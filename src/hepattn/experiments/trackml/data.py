from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset

HIT_COORDINATE_SCALE = 0.01


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class TrackMLDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_volume_ids: list | None = None,
        feature_volume_ids: dict | None = None,
        particle_min_pt: float = 1.0,
        particle_train_min_pt: float | None = None,
        particle_max_abs_eta: float = 2.5,
        particle_min_num_hits: int = 3,
        particle_train_min_num_hits: int | None = None,
        event_max_num_particles: int = 1000,
        strict_max_objects: bool = False,
        hit_eval_path: str | None = None,
        hit_filter_threshold: float = 0.1,
        include_paper_eval_targets: bool = False,
        dummy_data: bool = False,
    ):
        super().__init__()

        # Store dummy_data flag
        self.dummy_data = dummy_data

        # Set the global random sampling seed
        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        self.eval_particle_min_pt = float(particle_min_pt)
        self.train_particle_min_pt = float(particle_train_min_pt if particle_train_min_pt is not None else particle_min_pt)
        self.particle_max_abs_eta = float(particle_max_abs_eta)
        self.eval_particle_min_num_hits = int(particle_min_num_hits)
        self.train_particle_min_num_hits = int(
            particle_train_min_num_hits if particle_train_min_num_hits is not None else particle_min_num_hits
        )
        self.include_paper_eval_targets = bool(include_paper_eval_targets)
        self.has_distinct_training_particle_cuts = (
            self.train_particle_min_pt != self.eval_particle_min_pt
            or self.train_particle_min_num_hits != self.eval_particle_min_num_hits
        )

        if self.train_particle_min_pt > self.eval_particle_min_pt:
            msg = (
                "particle_train_min_pt must be less than or equal to particle_min_pt so the "
                "training truth remains a superset of the eval truth."
            )
            raise ValueError(msg)
        if self.train_particle_min_num_hits > self.eval_particle_min_num_hits:
            msg = (
                "particle_train_min_num_hits must be less than or equal to particle_min_num_hits so the "
                "training truth remains a superset of the eval truth."
            )
            raise ValueError(msg)

        # If using dummy data, skip file-based initialization
        if self.dummy_data:
            rank_zero_info("Generating dummy data...")
            self.dirpath = Path(dirpath) if dirpath else Path()
            self.hit_eval_path = None
            self.hit_filter_threshold = hit_filter_threshold
            self.inputs = inputs
            self.targets = targets
            self.num_events = max(num_events, 1) if num_events > 0 else 10
            self.event_names = [f"dummy_event_{i:06d}" for i in range(self.num_events)]
            self.sample_ids = list(range(self.num_events))
            self.hit_volume_ids = hit_volume_ids
            self.event_max_num_particles = event_max_num_particles
            return

        # Get a list of event names
        event_names = [Path(file).stem.replace("-parts", "") for file in Path(dirpath).glob("event*-parts.parquet")]

        # Calculate the number of events that will actually be used
        num_events_available = len(event_names)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)

        if num_events < 0:
            num_events = num_events_available

        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        # Metadata
        self.dirpath = Path(dirpath)
        self.hit_eval_path = hit_eval_path
        self.hit_filter_threshold = hit_filter_threshold
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]
        self.sample_ids = [int(name.removeprefix("event")) for name in self.event_names]

        # Setup hit eval file if specified
        if self.hit_eval_path:
            rank_zero_info(f"Using hit eval dataset {self.hit_eval_path}")

        # Hit level cuts
        self.hit_volume_ids = hit_volume_ids
        # Optional per-feature hit volume selections
        self.feature_volume_ids = feature_volume_ids

        # Particle level cuts
        self.particle_min_pt = self.eval_particle_min_pt
        self.particle_min_num_hits = self.eval_particle_min_num_hits

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles
        self.strict_max_objects = strict_max_objects

    def __len__(self):
        return int(self.num_events)

    def _swap_target_view(self, targets: dict[str, torch.Tensor], base_key: str, eval_value: torch.Tensor) -> None:
        train_key = f"{base_key}_train"
        eval_key = f"{base_key}_eval"
        targets[train_key] = targets[base_key].clone()
        targets[eval_key] = eval_value

    def _add_hit_targets(self, targets: dict[str, torch.Tensor], target_feature: str, fields: list[str], hits: pd.DataFrame) -> None:
        if "on_valid_particle" in fields:
            train_value = torch.from_numpy(hits["on_valid_particle"].to_numpy(copy=True)).unsqueeze(0)
            targets[f"{target_feature}_on_valid_particle"] = train_value
            if self.has_distinct_training_particle_cuts:
                eval_value = torch.from_numpy(hits["on_valid_particle_eval"].to_numpy(copy=True)).unsqueeze(0)
                self._swap_target_view(targets, f"{target_feature}_on_valid_particle", eval_value)

        if "is_first" in fields:
            train_value = torch.from_numpy(hits["is_first"].to_numpy(copy=True)).unsqueeze(0)
            targets[f"{target_feature}_is_first"] = train_value
            if self.has_distinct_training_particle_cuts:
                eval_value = torch.from_numpy(hits["is_first_eval"].to_numpy(copy=True)).unsqueeze(0)
                self._swap_target_view(targets, f"{target_feature}_is_first", eval_value)

        if "is_last" in fields:
            train_value = torch.from_numpy(hits["is_last"].to_numpy(copy=True)).unsqueeze(0)
            targets[f"{target_feature}_is_last"] = train_value
            if self.has_distinct_training_particle_cuts:
                eval_value = torch.from_numpy(hits["is_last_eval"].to_numpy(copy=True)).unsqueeze(0)
                self._swap_target_view(targets, f"{target_feature}_is_last", eval_value)

    def __getitem__(self, idx):
        if self.dummy_data:
            return self._generate_dummy_data(idx)

        inputs = {}
        targets = {}

        # Load the event
        hits, particles, paper_parts, truth = self.load_event(idx)
        num_particles = len(particles)

        # Build the input hits
        for feature, fields in self.inputs.items():
            # Determine per-feature hit subset
            if self.feature_volume_ids is not None and feature in self.feature_volume_ids:
                feature_hits = hits[hits["volume_id"].isin(self.feature_volume_ids[feature])]
            else:
                feature_hits = hits

            # Valid mask is all True for the feature-specific subset
            inputs[f"{feature}_valid"] = torch.full((len(feature_hits),), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(feature_hits[field].to_numpy(copy=True)).unsqueeze(0).half()

        # Create the targets for whether a particle slot is used or not
        if num_particles > self.event_max_num_particles:
            if self.strict_max_objects:
                message = f"Event {idx} has {num_particles}, but limit is {self.event_max_num_particles}"
                raise ValueError(message)
            particles = particles.iloc[: self.event_max_num_particles]
            num_particles = self.event_max_num_particles

        # Create particle_valid mask by concatenating True and False arrays
        num_padding = self.event_max_num_particles - num_particles
        targets["particle_valid"] = torch.cat([torch.full((num_particles,), True), torch.full((num_padding,), False)]).unsqueeze(0)
        if self.has_distinct_training_particle_cuts:
            eval_particle_valid = torch.cat(
                [
                    torch.from_numpy(particles["is_eval_particle"].to_numpy(copy=True)),
                    torch.full((num_padding,), False),
                ]
            ).unsqueeze(0)
            self._swap_target_view(targets, "particle_valid", eval_particle_valid)

        # Create the mask targets
        selected_particle_ids = torch.from_numpy(particles["particle_id"].to_numpy(copy=True))
        particle_ids = torch.cat([selected_particle_ids, torch.full((num_padding,), -999)])
        hit_particle_ids = torch.from_numpy(hits["particle_id"].to_numpy(copy=True))
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)
        if self.has_distinct_training_particle_cuts:
            eval_particle_hit_valid = targets["particle_hit_valid"] & targets["particle_valid_eval"].unsqueeze(-1)
            self._swap_target_view(targets, "particle_hit_valid", eval_particle_hit_valid)

        # Store particle and hit IDs for dynamic query selection
        targets["hit_particle_id"] = hit_particle_ids.unsqueeze(0)  # (1, N_hits)
        targets["particle_id"] = particle_ids.unsqueeze(0)  # (1, N_particles)

        # Create the hit filter targets (note this ignores the event_max_num_particles filtering)
        for target_feature, fields in self.targets.items():
            self._add_hit_targets(targets=targets, target_feature=target_feature, fields=fields, hits=hits)

        # Add sample ID
        targets["sample_id"] = torch.tensor([self.sample_ids[idx]], dtype=torch.int32)

        if self.include_paper_eval_targets:
            paper_vz = paper_parts["vz"].to_numpy(copy=True) if "vz" in paper_parts.columns else np.full(len(paper_parts), np.nan, dtype=np.float32)
            targets["paper_hit_particle_id"] = torch.from_numpy(hits["particle_id"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_hit_id"] = torch.from_numpy(hits["hit_id"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_hit_weight"] = torch.from_numpy(hits["weight"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_truth_particle_id"] = torch.from_numpy(truth["particle_id"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_truth_hit_id"] = torch.from_numpy(truth["hit_id"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_truth_weight"] = torch.from_numpy(truth["weight"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_all_particle_id"] = torch.from_numpy(paper_parts["particle_id"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_all_particle_pt"] = torch.from_numpy(paper_parts["pt"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_all_particle_eta"] = torch.from_numpy(paper_parts["eta"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_all_particle_phi"] = torch.from_numpy(paper_parts["phi"].to_numpy(copy=True)).unsqueeze(0)
            targets["paper_all_particle_vz"] = torch.from_numpy(paper_vz).unsqueeze(0)
            targets["paper_all_particle_n_hits"] = torch.from_numpy(paper_parts["n_hits"].to_numpy(copy=True)).unsqueeze(0)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/particle slots are filled with nans
                # This acts as a sanity check that we correctly mask out null slots in the loss
                x = torch.full((self.event_max_num_particles,), torch.nan)
                x[:num_particles] = torch.from_numpy(particles[field].to_numpy(copy=True)[: self.event_max_num_particles])
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets

    def load_event(self, idx):
        event_name = self.event_names[idx]

        particles = pd.read_parquet(self.dirpath / Path(event_name + "-parts.parquet"))
        hits = pd.read_parquet(self.dirpath / Path(event_name + "-hits.parquet"))

        # Make the detector volume selection
        if self.hit_volume_ids:
            hits = hits[hits["volume_id"].isin(self.hit_volume_ids)]

        # Scale the input coordinates to in meters so they are ~ 1
        for coord in ["x", "y", "z"]:
            hits[coord] *= HIT_COORDINATE_SCALE

        # Add extra hit fields
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        hits["theta"] = np.arccos(hits["z"] / hits["s"])
        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)

        # Add extra particle fields
        particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        particles["qopt"] = particles["q"] / particles["pt"]
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["theta"] = np.arccos(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["costheta"] = np.cos(particles["theta"])
        particles["sintheta"] = np.sin(particles["theta"])
        particles["cosphi"] = np.cos(particles["phi"])
        particles["sinphi"] = np.sin(particles["phi"])
        paper_parts = particles.copy()
        prefilter_hit_counts = hits["particle_id"].value_counts()
        paper_parts["n_hits"] = paper_parts["particle_id"].map(prefilter_hit_counts).fillna(0).astype(np.int64)

        # If a hit eval file was specified, read in the predictions from it to use the hit filtering
        if self.hit_eval_path:
            with h5py.File(self.hit_eval_path, "r") as hit_eval_file:
                if f"{self.sample_ids[idx]}/preds/final/hit_filter/hit_on_valid_particle_prob" in hit_eval_file:
                    hit_filter_probs = hit_eval_file[f"{self.sample_ids[idx]}/preds/final/hit_filter/hit_on_valid_particle_prob"][0]
                    hit_filter_pred = hit_filter_probs >= self.hit_filter_threshold
                elif f"{self.sample_ids[idx]}/preds/final/hit_filter/key_on_valid_particle_prob" in hit_eval_file:
                    hit_filter_probs = hit_eval_file[f"{self.sample_ids[idx]}/preds/final/hit_filter/key_on_valid_particle_prob"][0]
                    hit_filter_pred = hit_filter_probs >= self.hit_filter_threshold
                else:
                    hit_filter_pred = hit_eval_file[f"{self.sample_ids[idx]}/preds/final/hit_filter/hit_on_valid_particle"][0]
                hits = hits[hit_filter_pred]

        # TODO: Add back truth based hit filtering
        truth = hits[["hit_id", "particle_id", "weight"]].copy()

        counts = hits["particle_id"].value_counts()
        keep_particle_ids_train = counts[counts >= self.train_particle_min_num_hits].index.to_numpy()
        keep_particle_ids_eval = counts[counts >= self.eval_particle_min_num_hits].index.to_numpy()

        train_particle_mask = (particles["pt"] > self.train_particle_min_pt) & (particles["eta"].abs() < self.particle_max_abs_eta)
        eval_particle_mask = (particles["pt"] > self.eval_particle_min_pt) & (particles["eta"].abs() < self.particle_max_abs_eta)

        particles_train = particles[train_particle_mask & particles["particle_id"].isin(keep_particle_ids_train)].copy()
        eval_particle_ids = particles.loc[
            eval_particle_mask & particles["particle_id"].isin(keep_particle_ids_eval),
            "particle_id",
        ].to_numpy(copy=True)
        particles_train["is_eval_particle"] = particles_train["particle_id"].isin(eval_particle_ids)

        # Keep the strict eval particles first so loosening the training cuts does not evict them when the
        # event is capped to event_max_num_particles.
        particles_eval = particles_train[particles_train["is_eval_particle"]]
        particles_train_only = particles_train[~particles_train["is_eval_particle"]]
        particles = pd.concat([particles_eval, particles_train_only], ignore_index=True)

        # Mark which hits are on a valid / reconstructable particle, for the hit filter
        hits["on_valid_particle"] = hits["particle_id"].isin(particles["particle_id"])
        hits["on_valid_particle_eval"] = hits["particle_id"].isin(eval_particle_ids)

        # Mark which hits are the first (innermost) and last (outermost) hit on each particle track
        # Only calculate for hits on valid particles
        valid_hits = hits[hits["on_valid_particle"]]
        first_hit_indices = valid_hits.groupby("particle_id")["r"].idxmin()
        hits["is_first"] = False
        hits.loc[first_hit_indices, "is_first"] = True
        last_hit_indices = valid_hits.groupby("particle_id")["r"].idxmax()
        hits["is_last"] = False
        hits.loc[last_hit_indices, "is_last"] = True

        valid_hits_eval = hits[hits["on_valid_particle_eval"]]
        first_hit_indices_eval = valid_hits_eval.groupby("particle_id")["r"].idxmin()
        hits["is_first_eval"] = False
        hits.loc[first_hit_indices_eval, "is_first_eval"] = True
        last_hit_indices_eval = valid_hits_eval.groupby("particle_id")["r"].idxmax()
        hits["is_last_eval"] = False
        hits.loc[last_hit_indices_eval, "is_last_eval"] = True

        # Sanity checks
        assert len(particles) != 0, "No particles remaining - loosen selection!"
        assert len(hits) != 0, "No hits remaining - loosen selection!"
        assert particles["particle_id"].nunique() == len(particles), "Non-unique particle ids"

        # Check that all hits have different phi
        # This is necessary as the fast sorting algorithm used by pytorch can be non-stable
        # if two values are equal, which could cause subtle bugs
        # msg = f"Only {hits['phi'].nunique()} of the {len(hits)} have unique phi"
        # assert hits["phi"].nunique() == len(hits), msg

        return hits, particles, paper_parts, truth

    def _generate_dummy_data(self, idx):
        """Generate completely random dummy data for CI testing."""
        inputs = {}
        targets = {}

        # Create random number generator
        rng = np.random.default_rng(self.sampling_seed + idx)

        # Generate random number of hits (between 10 and 100)
        num_hits = rng.integers(10, 101)

        # Generate random number of particles (up to event_max_num_particles)
        num_particles = rng.integers(1, min(self.event_max_num_particles + 1, 101))

        # Build the input hits with random data
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                # Generate random normal data for all fields
                data = rng.standard_normal(num_hits)
                inputs[f"{feature}_{field}"] = torch.from_numpy(data).unsqueeze(0).to(torch.float32)

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_particles] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)
        if self.has_distinct_training_particle_cuts:
            num_eval_particles = max(1, num_particles // 2)
            eval_particle_valid = torch.full((self.event_max_num_particles,), False)
            eval_particle_valid[:num_eval_particles] = True
            self._swap_target_view(targets, "particle_valid", eval_particle_valid.unsqueeze(0))

        # Build dummy particle IDs
        particle_ids = torch.arange(num_particles, dtype=torch.long)
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - num_particles)])

        # Assign random particle IDs to hits
        hit_particle_ids = torch.randint(0, num_particles, (num_hits,))

        # Create the mask targets
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)
        if self.has_distinct_training_particle_cuts:
            eval_particle_hit_valid = targets["particle_hit_valid"] & targets["particle_valid_eval"].unsqueeze(-1)
            self._swap_target_view(targets, "particle_hit_valid", eval_particle_hit_valid)

        # Store particle and hit IDs for dynamic query selection
        targets["hit_particle_id"] = hit_particle_ids.unsqueeze(0)  # (1, N_hits)
        targets["particle_id"] = particle_ids.unsqueeze(0)  # (1, N_particles)

        hit_on_valid_particle = torch.from_numpy((hit_particle_ids.numpy() < num_particles)).unsqueeze(0)
        targets["hit_on_valid_particle"] = hit_on_valid_particle
        targets["hit_is_first"] = torch.randint(0, 2, (num_hits,), dtype=torch.bool).unsqueeze(0)
        targets["hit_is_last"] = torch.randint(0, 2, (num_hits,), dtype=torch.bool).unsqueeze(0)
        if self.has_distinct_training_particle_cuts:
            eval_hit_on_valid_particle = targets["particle_valid_eval"][0, hit_particle_ids].unsqueeze(0)
            self._swap_target_view(targets, "hit_on_valid_particle", eval_hit_on_valid_particle)
            self._swap_target_view(targets, "hit_is_first", targets["hit_is_first"] & eval_hit_on_valid_particle)
            self._swap_target_view(targets, "hit_is_last", targets["hit_is_last"] & eval_hit_on_valid_particle)

        # Add sample ID
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Generate random particle data
                x = torch.full((self.event_max_num_particles,), torch.nan)
                data = rng.standard_normal(num_particles)
                x[:num_particles] = torch.from_numpy(data)
                targets[f"particle_{field}"] = x.unsqueeze(0)

        if self.include_paper_eval_targets:
            hit_indices = torch.arange(num_hits, dtype=torch.int64)
            hit_weights = torch.full((num_hits,), 1.0 / max(num_hits, 1), dtype=torch.float32)
            particle_eta = torch.full((num_particles,), torch.nan)
            particle_phi = torch.full((num_particles,), torch.nan)
            particle_vz = torch.full((num_particles,), torch.nan)
            particle_pt = torch.full((num_particles,), torch.nan)
            if "particle" in self.targets:
                if "eta" in self.targets["particle"]:
                    particle_eta[:num_particles] = targets["particle_eta"][0, :num_particles]
                if "phi" in self.targets["particle"]:
                    particle_phi[:num_particles] = targets["particle_phi"][0, :num_particles]
                if "pt" in self.targets["particle"]:
                    particle_pt[:num_particles] = targets["particle_pt"][0, :num_particles]

            targets["paper_hit_particle_id"] = hit_particle_ids.unsqueeze(0)
            targets["paper_hit_id"] = hit_indices.unsqueeze(0)
            targets["paper_hit_weight"] = hit_weights.unsqueeze(0)
            targets["paper_truth_particle_id"] = hit_particle_ids.unsqueeze(0)
            targets["paper_truth_hit_id"] = hit_indices.unsqueeze(0)
            targets["paper_truth_weight"] = hit_weights.unsqueeze(0)
            targets["paper_all_particle_id"] = particle_ids[:num_particles].unsqueeze(0)
            targets["paper_all_particle_pt"] = particle_pt.unsqueeze(0)
            targets["paper_all_particle_eta"] = particle_eta.unsqueeze(0)
            targets["paper_all_particle_phi"] = particle_phi.unsqueeze(0)
            targets["paper_all_particle_vz"] = particle_vz.unsqueeze(0)
            particle_n_hits = torch.bincount(hit_particle_ids, minlength=max(num_particles, 1))[:num_particles].to(torch.int64)
            targets["paper_all_particle_n_hits"] = particle_n_hits.unsqueeze(0)

        return inputs, targets


class TrackMLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        paper_compatible_test_output: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.paper_compatible_test_output = paper_compatible_test_output
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dataset = TrackMLDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                hit_eval_path=self.hit_eval_train,
                include_paper_eval_targets=False,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dataset = TrackMLDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                hit_eval_path=self.hit_eval_val,
                include_paper_eval_targets=False,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit":
            rank_zero_info(f"Created training dataset with {len(self.train_dataset):,} events")
            rank_zero_info(f"Created validation dataset with {len(self.val_dataset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"

            self.test_dataset = TrackMLDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                hit_eval_path=self.hit_eval_test,
                include_paper_eval_targets=self.paper_compatible_test_output,
                **self.kwargs,
            )
            rank_zero_info(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: TrackMLDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)
