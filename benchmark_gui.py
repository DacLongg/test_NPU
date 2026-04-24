#!/usr/bin/env python3

import threading
import traceback


try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tkinter is not installed. On Ubuntu, install it with:\n"
        "  sudo apt install python3-tk\n"
        "Then run this script again."
    ) from exc

import TestNpu as bench


DEVICE_COLORS = {
    "CPU": "#e76f51",
    "GPU": "#2a9d8f",
    "NPU": "#264653",
}


class BenchmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CPU / GPU / NPU Benchmark")
        self.root.geometry("1380x640")
        self.root.configure(bg="#f4efe7")

        self.available_devices = []
        self.is_running = False
        self._build_ui()
        self._load_devices()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4efe7")
        style.configure("TLabelframe", background="#fffdf8", borderwidth=1)
        style.configure("TLabelframe.Label", background="#fffdf8", foreground="#1f2937")
        style.configure("TLabel", background="#f4efe7", foreground="#1f2937")
        style.configure("Header.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("Muted.TLabel", foreground="#6b7280")
        style.configure("Treeview", rowheight=26)
        style.configure("TButton", padding=8)

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="CPU / GPU / NPU Benchmark", style="Header.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            header,
            text="Chon workload, bam Start Benchmark, va xem ket qua so sanh truc quan.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(16, 0))

        left = ttk.LabelFrame(top, text="Benchmark Options", padding=14)
        left.pack(side="left", fill="x", expand=True)

        right = ttk.LabelFrame(top, text="Run Status", padding=14)
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.model_type_var = tk.StringVar(value="cnn")
        self.hint_var = tk.StringVar(value="THROUGHPUT")
        self.dtype_var = tk.StringVar(value="fp16")
        self.batch_var = tk.StringVar(value="4")
        self.height_var = tk.StringVar(value="224")
        self.width_var = tk.StringVar(value="224")
        self.channels_var = tk.StringVar(value="3")
        self.features_var = tk.StringVar(value="1024")
        self.layers_var = tk.StringVar(value="6")
        self.warmup_var = tk.StringVar(value="20")
        self.iterations_var = tk.StringVar(value="100")
        self.requests_var = tk.StringVar(value="4")
        self.repeats_var = tk.StringVar(value="3")

        self.device_vars = {
            "CPU": tk.BooleanVar(value=True),
            "GPU": tk.BooleanVar(value=True),
            "NPU": tk.BooleanVar(value=True),
        }

        fields = ttk.Frame(left)
        fields.pack(fill="x")

        self._add_inline_row(
            fields,
            0,
            [
                ("Model", lambda parent: self._make_combo(parent, self.model_type_var, ["cnn", "mlp"], width=12)),
                ("Hint", lambda parent: self._make_combo(parent, self.hint_var, ["LATENCY", "THROUGHPUT"], width=14)),
                ("DType", lambda parent: self._make_combo(parent, self.dtype_var, ["fp16", "fp32"], width=10)),
                ("Batch", lambda parent: ttk.Entry(parent, textvariable=self.batch_var, width=10)),
            ],
        )
        self._add_inline_row(
            fields,
            1,
            [
                ("Warmup", lambda parent: ttk.Entry(parent, textvariable=self.warmup_var, width=10)),
                ("Iterations", lambda parent: ttk.Entry(parent, textvariable=self.iterations_var, width=10)),
                ("Requests", lambda parent: ttk.Entry(parent, textvariable=self.requests_var, width=10)),
                ("Repeats", lambda parent: ttk.Entry(parent, textvariable=self.repeats_var, width=10)),
            ],
        )
        self._add_inline_row(
            fields,
            2,
            [
                ("Image H", lambda parent: ttk.Entry(parent, textvariable=self.height_var, width=10)),
                ("Image W", lambda parent: ttk.Entry(parent, textvariable=self.width_var, width=10)),
                ("Channels", lambda parent: ttk.Entry(parent, textvariable=self.channels_var, width=10)),
                ("Features", lambda parent: ttk.Entry(parent, textvariable=self.features_var, width=10)),
                ("Layers", lambda parent: ttk.Entry(parent, textvariable=self.layers_var, width=10)),
            ],
        )

        devices_frame = ttk.Frame(left)
        devices_frame.pack(anchor="w", pady=(10, 0))
        ttk.Label(devices_frame, text="Devices").pack(anchor="w")
        for name in ["CPU", "GPU", "NPU"]:
            ttk.Checkbutton(devices_frame, text=name, variable=self.device_vars[name]).pack(
                side="left", padx=(0, 12)
            )

        self.start_button = ttk.Button(
            right, text="Start Benchmark", command=self.start_benchmark
        )
        self.start_button.pack(anchor="w")

        self.refresh_button = ttk.Button(
            right, text="Refresh Devices", command=self._load_devices
        )
        self.refresh_button.pack(anchor="w", pady=(10, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(right, textvariable=self.status_var, style="Muted.TLabel", wraplength=420).pack(
            anchor="w", pady=(16, 0)
        )

        self.device_info = tk.Text(
            right,
            height=8,
            width=54,
            bg="#fffdf8",
            fg="#1f2937",
            relief="solid",
            borderwidth=1,
        )
        self.device_info.pack(fill="both", expand=True, pady=(12, 0))

        bottom = ttk.Frame(outer)
        bottom.pack(fill="both", expand=True, pady=(18, 0))

        charts = ttk.LabelFrame(bottom, text="Visualization", padding=14)
        charts.pack(side="left", fill="both", expand=True)

        table_frame = ttk.LabelFrame(bottom, text="Detailed Results", padding=14)
        table_frame.pack(side="left", fill="both", expand=True, padx=(16, 0))

        self.canvas = tk.Canvas(
            charts,
            width=520,
            height=360,
            bg="#fffdf8",
            highlightthickness=1,
            highlightbackground="#e7dccb",
        )
        self.canvas.pack(fill="both", expand=True)

        columns = ("device", "latency", "throughput", "hint", "requests")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.tree.heading("device", text="Device")
        self.tree.heading("latency", text="Median Latency (ms)")
        self.tree.heading("throughput", text="Median Throughput (FPS)")
        self.tree.heading("hint", text="Hint")
        self.tree.heading("requests", text="Requests")
        self.tree.column("device", width=80, anchor="center")
        self.tree.column("latency", width=130, anchor="center")
        self.tree.column("throughput", width=140, anchor="center")
        self.tree.column("hint", width=100, anchor="center")
        self.tree.column("requests", width=80, anchor="center")
        self.tree.pack(fill="both", expand=True)

        self.summary_var = tk.StringVar(value="No benchmark result yet.")
        ttk.Label(table_frame, textvariable=self.summary_var, style="Muted.TLabel", wraplength=420).pack(
            anchor="w", pady=(10, 0)
        )

        explain_frame = ttk.LabelFrame(table_frame, text="Interpretation", padding=10)
        explain_frame.pack(fill="x", pady=(12, 0))
        self.explain_text = tk.Text(
            explain_frame,
            height=18,
            wrap="word",
            bg="#fffdf8",
            fg="#1f2937",
            relief="solid",
            borderwidth=1,
        )
        explain_scroll = ttk.Scrollbar(explain_frame, orient="vertical", command=self.explain_text.yview)
        self.explain_text.configure(yscrollcommand=explain_scroll.set)
        self.explain_text.pack(side="left", fill="both", expand=True)
        explain_scroll.pack(side="right", fill="y")
        self._set_explanation(
            "Run a benchmark to generate an automatic explanation and conclusion."
        )

    def _add_inline_row(self, parent, row, items):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="w", pady=4)
        for index, (label, widget_factory) in enumerate(items):
            field = ttk.Frame(frame)
            field.grid(row=0, column=index, sticky="w", padx=(0, 14))
            ttk.Label(field, text=label).pack(anchor="w")
            widget = widget_factory(field)
            widget.pack(anchor="w", pady=(2, 0))

    def _make_combo(self, parent, variable, values, width=16):
        combo = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state="readonly")
        return combo

    def _load_devices(self):
        try:
            _, _, Core, _, _ = bench.import_dependencies()
            core = Core()
            self.available_devices = list(core.available_devices)
            lines = ["Detected devices: {}".format(self.available_devices)]
            for device in self.available_devices:
                try:
                    name = core.get_property(device, "FULL_DEVICE_NAME")
                except Exception as exc:
                    name = "Unavailable ({})".format(exc)
                lines.append("- {}: {}".format(device, name))
            self.device_info.delete("1.0", tk.END)
            self.device_info.insert("1.0", "\n".join(lines))
            self.status_var.set("Device list refreshed.")
        except Exception as exc:
            self.device_info.delete("1.0", tk.END)
            self.device_info.insert("1.0", "Failed to load devices:\n{}".format(exc))
            self.status_var.set("Failed to refresh devices.")

    def _collect_config(self):
        selected_devices = [
            name
            for name, variable in self.device_vars.items()
            if variable.get() and name in self.available_devices
        ]
        if not selected_devices:
            raise ValueError("No available device selected.")

        return {
            "model_type": self.model_type_var.get(),
            "hint": self.hint_var.get(),
            "dtype": self.dtype_var.get(),
            "batch": int(self.batch_var.get()),
            "height": int(self.height_var.get()),
            "width": int(self.width_var.get()),
            "channels": int(self.channels_var.get()),
            "features": int(self.features_var.get()),
            "layers": int(self.layers_var.get()),
            "warmup": int(self.warmup_var.get()),
            "iterations": int(self.iterations_var.get()),
            "num_requests": int(self.requests_var.get()),
            "repeats": int(self.repeats_var.get()),
            "devices": selected_devices,
        }

    def start_benchmark(self):
        if self.is_running:
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.is_running = True
        self.start_button.configure(state="disabled")
        self.status_var.set("Benchmark is running...")
        self.summary_var.set("Running benchmark...")
        self._clear_results()

        thread = threading.Thread(target=self._run_benchmark_worker, args=(config,), daemon=True)
        thread.start()

    def _run_benchmark_worker(self, config):
        try:
            np, _, Core, Model, opset8 = bench.import_dependencies()
            core = Core()
            model = bench.create_benchmark_model(
                np=np,
                Model=Model,
                opset8=opset8,
                model_type=config["model_type"],
                batch=config["batch"],
                features=config["features"],
                layers=config["layers"],
                channels=config["channels"],
                height=config["height"],
                width=config["width"],
                dtype=config["dtype"],
            )

            results = []
            failures = []
            for device in config["devices"]:
                try:
                    result = bench.benchmark_device(
                        core=core,
                        np=np,
                        model=model,
                        device=device,
                        iterations=config["iterations"],
                        warmup=config["warmup"],
                        model_type=config["model_type"],
                        batch=config["batch"],
                        features=config["features"],
                        channels=config["channels"],
                        height=config["height"],
                        width=config["width"],
                        dtype=config["dtype"],
                        hint=config["hint"],
                        num_requests=config["num_requests"],
                        repeats=config["repeats"],
                    )
                    results.append(result)
                except Exception as exc:
                    failures.append("{}: {}".format(device, exc))

            self.root.after(0, self._on_benchmark_done, config, results, failures)
        except Exception:
            self.root.after(0, self._on_benchmark_crash, traceback.format_exc())

    def _on_benchmark_done(self, config, results, failures):
        self.is_running = False
        self.start_button.configure(state="normal")

        if not results:
            self.status_var.set("Benchmark failed on all selected devices.")
            detail = "\n".join(failures) if failures else "Unknown error."
            messagebox.showerror("Benchmark failed", detail)
            return

        self.status_var.set("Benchmark completed.")
        self._render_results(results)

        summary_lines = [
            "Model: {} | Hint: {} | Batch: {} | Requests: {}".format(
                config["model_type"], config["hint"], config["batch"], config["num_requests"]
            )
        ]
        if failures:
            summary_lines.append("Failed devices: {}".format("; ".join(failures)))

        best_latency = min(results, key=lambda item: item["avg_latency_ms"])
        best_throughput = max(results, key=lambda item: item["throughput_fps"])
        summary_lines.append(
            "Best latency: {} ({:.3f} ms)".format(
                best_latency["device"], best_latency["avg_latency_ms"]
            )
        )
        summary_lines.append(
            "Best throughput: {} ({:.2f} FPS)".format(
                best_throughput["device"], best_throughput["throughput_fps"]
            )
        )
        self.summary_var.set(" | ".join(summary_lines))
        self._set_explanation(self._build_interpretation(config, results, failures))

    def _on_benchmark_crash(self, error_text):
        self.is_running = False
        self.start_button.configure(state="normal")
        self.status_var.set("Benchmark crashed.")
        messagebox.showerror("Benchmark crashed", error_text)

    def _clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.canvas.delete("all")
        self._set_explanation(
            "Run a benchmark to generate an automatic explanation and conclusion."
        )

    def _render_results(self, results):
        self._clear_results()
        for result in results:
            self.tree.insert(
                "",
                "end",
                values=(
                    result["device"],
                    "{:.3f}".format(result["avg_latency_ms"]),
                    "{:.2f}".format(result["throughput_fps"]),
                    result["hint"],
                    result["num_requests"],
                ),
            )
        self._draw_chart(results)

    def _set_explanation(self, text):
        self.explain_text.configure(state="normal")
        self.explain_text.delete("1.0", tk.END)
        self.explain_text.insert("1.0", text)
        self.explain_text.configure(state="disabled")

    def _build_interpretation(self, config, results, failures):
        lines = []
        if config["model_type"] == "cnn":
            workload = "synthetic CNN, batch {}, input {}x{}x{}".format(
                config["batch"], config["channels"], config["height"], config["width"]
            )
        else:
            workload = "synthetic MLP, batch {}, features {}, layers {}".format(
                config["batch"], config["features"], config["layers"]
            )

        lines.append(
            "This run used {} with {} mode and {} parallel request(s).".format(
                workload, config["hint"], config["num_requests"]
            )
        )

        best_latency = min(results, key=lambda item: item["avg_latency_ms"])
        best_throughput = max(results, key=lambda item: item["throughput_fps"])
        lines.append(
            "Best latency belongs to {} at {:.3f} ms. Best throughput belongs to {} at {:.2f} FPS.".format(
                best_latency["device"],
                best_latency["avg_latency_ms"],
                best_throughput["device"],
                best_throughput["throughput_fps"],
            )
        )

        ordered_latency = sorted(results, key=lambda item: item["avg_latency_ms"])
        ordered_throughput = sorted(results, key=lambda item: item["throughput_fps"], reverse=True)
        lines.append(
            "Latency ranking: {}. Throughput ranking: {}.".format(
                " < ".join(result["device"] for result in ordered_latency),
                " > ".join(result["device"] for result in ordered_throughput),
            )
        )

        by_device = {result["device"]: result for result in results}
        if {"CPU", "GPU", "NPU"}.issubset(by_device):
            cpu = by_device["CPU"]
            gpu = by_device["GPU"]
            npu = by_device["NPU"]
            lines.append(
                "Compared with CPU, GPU is {:.2f}x faster on latency and {:.2f}x higher on throughput. "
                "NPU is {:.2f}x faster on latency and {:.2f}x higher on throughput.".format(
                    cpu["avg_latency_ms"] / gpu["avg_latency_ms"],
                    gpu["throughput_fps"] / cpu["throughput_fps"],
                    cpu["avg_latency_ms"] / npu["avg_latency_ms"],
                    npu["throughput_fps"] / cpu["throughput_fps"],
                )
            )

        if config["hint"] == "THROUGHPUT":
            lines.append(
                "Because this run favors throughput and uses multiple requests, the result highlights which device handles parallel inference most efficiently. "
                "That usually benefits GPU or NPU more than CPU."
            )
        else:
            lines.append(
                "Because this run uses latency mode, the result reflects single-request responsiveness more than maximum total throughput."
            )

        if config["model_type"] == "cnn":
            lines.append(
                "This is still a synthetic CNN, so the conclusion applies to this benchmark workload, not automatically to every real vision model."
            )
        else:
            lines.append(
                "This is a synthetic dense-tensor MLP workload, so the conclusion applies to matrix-heavy execution patterns more than to every real AI model."
            )

        primary_winner = best_throughput if config["hint"] == "THROUGHPUT" else best_latency
        lines.append(
            "Conclusion: for this exact input size, model type, and benchmark mode, {} is the best device overall.".format(
                primary_winner["device"]
            )
        )

        if failures:
            lines.append("Devices that failed in this run: {}.".format("; ".join(failures)))

        return "\n\n".join(lines)

    def _draw_chart(self, results):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 520)
        height = max(self.canvas.winfo_height(), 440)
        self.canvas.create_text(
            20, 20, anchor="nw", text="Latency vs Throughput", fill="#1f2937", font=("TkDefaultFont", 16, "bold")
        )
        self.canvas.create_text(
            20, 44,
            anchor="nw",
            text="Latency: lower is better | Throughput: higher is better",
            fill="#6b7280",
            font=("TkDefaultFont", 10),
        )

        latency_top = 90
        throughput_top = 270
        chart_left = 140
        chart_right = width - 40
        bar_height = 26
        bar_gap = 18

        max_latency = max(result["avg_latency_ms"] for result in results) or 1
        max_throughput = max(result["throughput_fps"] for result in results) or 1

        self.canvas.create_text(20, latency_top - 26, anchor="nw", text="Median Latency (ms)", fill="#1f2937", font=("TkDefaultFont", 12, "bold"))
        for index, result in enumerate(results):
            top = latency_top + index * (bar_height + bar_gap)
            bottom = top + bar_height
            width_ratio = result["avg_latency_ms"] / max_latency
            bar_width = max((chart_right - chart_left) * width_ratio, 6)
            color = DEVICE_COLORS.get(result["device"], "#c08457")
            self.canvas.create_text(20, top + 3, anchor="nw", text=result["device"], fill="#1f2937")
            self.canvas.create_rectangle(chart_left, top, chart_right, bottom, fill="#efe5d8", outline="")
            self.canvas.create_rectangle(chart_left, top, chart_left + bar_width, bottom, fill=color, outline="")
            self.canvas.create_text(chart_right, top + 3, anchor="ne", text="{:.3f} ms".format(result["avg_latency_ms"]), fill="#6b7280")

        self.canvas.create_text(20, throughput_top - 26, anchor="nw", text="Median Throughput (FPS)", fill="#1f2937", font=("TkDefaultFont", 12, "bold"))
        for index, result in enumerate(results):
            top = throughput_top + index * (bar_height + bar_gap)
            bottom = top + bar_height
            width_ratio = result["throughput_fps"] / max_throughput
            bar_width = max((chart_right - chart_left) * width_ratio, 6)
            color = DEVICE_COLORS.get(result["device"], "#c08457")
            self.canvas.create_text(20, top + 3, anchor="nw", text=result["device"], fill="#1f2937")
            self.canvas.create_rectangle(chart_left, top, chart_right, bottom, fill="#efe5d8", outline="")
            self.canvas.create_rectangle(chart_left, top, chart_left + bar_width, bottom, fill=color, outline="")
            self.canvas.create_text(chart_right, top + 3, anchor="ne", text="{:.2f} FPS".format(result["throughput_fps"]), fill="#6b7280")


def main():
    root = tk.Tk()
    app = BenchmarkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
