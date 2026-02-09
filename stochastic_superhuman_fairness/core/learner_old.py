class Learner_old:
    """
    Unified multi-phase learner that trains classifiers and RL/IL models
    using subdominance as the shared optimization signal.
    """

    def __init__(self, cfg, demonstrator, logger=None):
        self.cfg = normalize_cfg(cfg)
        self.demo = demonstrator
        self.logger = logger
        self.device = getattr(self.cfg.learner, "device", "cpu")

        self.model = None
        self.current_algo = None
        self.global_step = 0

        # Validate + normalize training schedule
        self.schedule = self._validate_schedule(self.cfg.learner.schedule)

    # ------------------------------------------------------------------
    def _validate_schedule(self, schedule):
        """Fill missing keys in schedule entries with defaults."""
        validated = []
        for i, phase in enumerate(schedule):
            entry = {**DEFAULT_PHASE_CFG, **phase}
            if "algo" not in entry:
                raise ValueError(f"Missing 'algo' in schedule entry {i}")
            entry["algo"] = entry["algo"].lower()
            validated.append(entry)
        return validated

    # ------------------------------------------------------------------
    def run_schedule(self):
        """Run sequential training phases defined in config."""
        for idx, phase_cfg in enumerate(self.schedule, 1):
            algo = phase_cfg["algo"]
            print(f"\n Phase {idx}: {algo.upper()}")
            self._run_phase(phase_cfg)
        print("\n✅ All schedule phases complete.")

    # ------------------------------------------------------------------
    def _run_phase(self, phase_cfg):
        algo = phase_cfg["algo"]
        epochs = phase_cfg["epochs"]
        batch_size = phase_cfg["batch_size"]

        self.switch_algo(algo, phase_cfg)
        print(f" Training {algo.upper()} for {epochs} epochs (lr={phase_cfg['lr']})...")

        for epoch in range(epochs):
            try:
                X, y, A = self.demo.step(
                    n_samples=batch_size,
                    flatten=True,
                    as_torch=True,
                    device=self.device,
                )
            except StopIteration:
                self.demo.reset()
                continue

            # === Logistic regression closed-form phase ===
            if algo == "logistic" and hasattr(self.model, "closed_form_fit"):
                loss_dict = self.model.closed_form_fit(X, y)
            else:
                # === General subdominance-based optimization ===
                loss_dict = self.model.train_step(X, y)

                # Compute subdominance loss & propagate through both nets
                subdom_loss = subdominance.compute_subdominance_loss(X, y, A, self.model)
                if hasattr(subdom_loss, "backward"):  # torch tensor
                    self.model.opt.zero_grad(set_to_none=True)
                    subdom_loss.backward()
                    self.model.opt.step()

                loss_dict["subdominance"] = float(subdom_loss.item() if hasattr(subdom_loss, "item") else subdom_loss)

                # Update value function if present
                if hasattr(self.model, "update_value"):
                    v_loss = self.model.update_value(X, y, A)
                    loss_dict["value_loss"] = float(v_loss)

            # Fairness signal (optional extra diagnostic)
            fairness_penalty = self._compute_fairness_loss(X, y, A)
            total_loss = loss_dict.get("loss", 0) + fairness_penalty + loss_dict.get("subdominance", 0)
            loss_dict["fairness_loss"] = fairness_penalty
            loss_dict["total_loss"] = total_loss

            metrics = {
                "algo": algo,
                "epoch": epoch,
                "global_step": self.global_step,
                **{k: float(v) for k, v in loss_dict.items()},
            }
            self._log(metrics)
            self.global_step += 1

        # evaluate at the end of each phase
        eval_acc = self.evaluate()
        self._log({"algo": algo, "eval_acc": eval_acc}, phase="eval")

    # ------------------------------------------------------------------
    def switch_algo(self, algo_name, phase_cfg):
        """Initialize a new model with parameter transfer."""
        algo_name = algo_name.lower()
        ModelClass = MODEL_REGISTRY.get(algo_name)
        if ModelClass is None:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        new_model = ModelClass(phase_cfg, self.demo)
        if hasattr(new_model, "to"):
            new_model.to(self.device)

        if self.model is not None:
            self._transfer_parameters(self.model, new_model)

        self.model = new_model
        self.current_algo = algo_name
        print(f"🔁 Switched to algorithm: {algo_name.upper()}")

    # ------------------------------------------------------------------
    def _transfer_parameters(self, old_model, new_model):
        """Transfer policy/value weights between compatible models."""
        if not hasattr(old_model, "get_state_dict") or not hasattr(new_model, "get_state_dict"):
            return
        old_sd, new_sd = old_model.get_state_dict(), new_model.get_state_dict()

        if "policy" in old_sd and "policy" in new_sd:
            self._safe_load(new_model.policy, old_sd["policy"], "policy")

        if "value" in old_sd and "value" in new_sd and old_sd["value"] and new_sd["value"]:
            self._safe_load(new_model.value, old_sd["value"], "value")

        print("🔄 Transferred compatible weights.")

    def _safe_load(self, module, state_dict, name):
        """Safely load overlapping parameters by name."""
        own_state = module.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in own_state and v.shape == own_state[k].shape}
        own_state.update(matched)
        module.load_state_dict(own_state)
        print(f"   → {len(matched)} {name} layers transferred")

    # ------------------------------------------------------------------
    def evaluate(self):
        """Evaluate current model on eval demos."""
        accs = []
        for demo_batch in self.demo.step(
            n_samples=len(self.demo.eval_demos), mode="eval", flatten=False
        ):
            X = torch.tensor(demo_batch["X"], dtype=torch.float32, device=self.device)
            y = torch.tensor(demo_batch["y"], dtype=torch.float32, device=self.device)
            y_pred = self.model.forward(X)
            acc = (torch.sigmoid(y_pred).round() == y).float().mean().item()
            accs.append(acc)

        avg = np.mean(accs)
        print(f"📊 Eval {self.current_algo}: acc={avg:.3f}")
        self.demo.reset("eval")
        return avg

    # ------------------------------------------------------------------
    def _compute_fairness_loss(self, X, y, A):
        """Optional fairness loss as diagnostic."""
        if hasattr(subdominance, "compute_fairness_penalty"):
            return subdominance.compute_fairness_penalty(X, y, A, self.model)
        return 0.0

    def _log(self, metrics, phase="train"):
        """Send metrics to logger or stdout."""
        if self.logger:
            self.logger.log(metrics, phase=phase)
        else:
            print(metrics)



