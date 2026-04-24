import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs NPU inference with OpenVINO."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured inference iterations per device.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations per device.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for the synthetic benchmark model.",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=1024,
        help="Input feature size for the synthetic benchmark model.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="Number of fully connected layers in the synthetic benchmark model.",
    )
    parser.add_argument(
        "--model-type",
        choices=["cnn", "mlp"],
        default="cnn",
        help="Synthetic model type. cnn is usually a better NPU smoke test.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input height for cnn model type.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input width for cnn model type.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Input channels for cnn model type.",
    )
    parser.add_argument(
        "--device",
        action="append",
        choices=["CPU", "GPU", "NPU"],
        help="Benchmark only the selected device(s). Can be provided multiple times.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Data type for benchmark tensors.",
    )
    parser.add_argument(
        "--hint",
        choices=["LATENCY", "THROUGHPUT"],
        default="LATENCY",
        help="OpenVINO performance hint.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Number of infer requests to run in parallel.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many benchmark runs to execute per device and compare by median.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip environment diagnostics and run benchmark directly.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run NPU environment probe instead of the full benchmark.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write benchmark results as JSON.",
    )
    return parser.parse_args()


def import_dependencies():
    missing = []
    openvino_module = None
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        import openvino as openvino_module  # type: ignore
        from openvino.runtime import Core, Model, opset8  # type: ignore
    except ImportError:
        Core = Model = opset8 = None
        missing.append("openvino")

    if missing:
        print("Missing Python packages: {}".format(", ".join(missing)))
        print("Install them first:")
        print("  python3 -m pip install numpy openvino")
        sys.exit(1)

    return np, openvino_module, Core, Model, opset8


def run_command(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
    except Exception as exc:
        return exc


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_diagnostics(core):
    print_header("OPENVINO DEVICE DIAGNOSTICS")

    print("1. Library Path")
    print("   LD_LIBRARY_PATH: {}".format(os.environ.get("LD_LIBRARY_PATH", "Not set")))

    print("\n2. Intel Level Zero Status")
    result = run_command(["find", "/usr", "-name", "libze_loader.so*"])
    if isinstance(result, Exception):
        print("   Could not inspect Level Zero libraries: {}".format(result))
    elif result.stdout.strip():
        print("   Found:")
        for line in result.stdout.strip().splitlines():
            print("   - {}".format(line))
    else:
        print("   Not found. On Ubuntu, install packages such as:")
        print("   - sudo apt install level-zero level-zero-dev")

    print("\n3. Intel Xe Driver")
    result = run_command(["lsmod"])
    if isinstance(result, Exception):
        print("   Could not inspect kernel modules: {}".format(result))
    elif "xe" in result.stdout:
        print("   xe driver is loaded")
    else:
        print("   xe driver is not loaded")

    print("\n4. Available OpenVINO Devices")
    devices = core.available_devices
    if not devices:
        print("   No OpenVINO device detected")
        return

    print("   Detected: {}".format(devices))
    for device in devices:
        try:
            full_name = core.get_property(device, "FULL_DEVICE_NAME")
        except Exception as exc:
            full_name = "Unavailable ({})".format(exc)
        print("   - {}: {}".format(device, full_name))


def create_mlp_model(np, Model, opset8, batch, features, layers, dtype):
    ov_dtype = np.float16 if dtype == "fp16" else np.float32
    rng = np.random.default_rng(42)

    input_node = opset8.parameter([batch, features], ov_dtype, name="input")
    node = input_node

    for _ in range(layers):
        weights = rng.standard_normal((features, features)).astype(ov_dtype)
        bias = rng.standard_normal((features,)).astype(ov_dtype)

        weight_node = opset8.constant(weights)
        bias_node = opset8.constant(bias)

        node = opset8.matmul(node, weight_node, False, False)
        node = opset8.add(node, bias_node)
        node = opset8.relu(node)

    return Model([node], [input_node], "cpu_vs_npu_benchmark")


def create_cnn_model(np, Model, opset8, batch, channels, height, width, dtype):
    ov_dtype = np.float16 if dtype == "fp16" else np.float32
    rng = np.random.default_rng(42)

    input_node = opset8.parameter([batch, channels, height, width], ov_dtype, name="input")
    node = input_node
    in_channels = channels

    for out_channels in [16, 32, 64]:
        weights = rng.standard_normal((out_channels, in_channels, 3, 3)).astype(ov_dtype)
        bias = rng.standard_normal((1, out_channels, 1, 1)).astype(ov_dtype)
        node = opset8.convolution(
            node,
            opset8.constant(weights),
            strides=[1, 1],
            pads_begin=[1, 1],
            pads_end=[1, 1],
            dilations=[1, 1],
        )
        node = opset8.add(node, opset8.constant(bias))
        node = opset8.relu(node)
        node = opset8.max_pool(
            node,
            strides=[2, 2],
            dilations=[1, 1],
            pads_begin=[0, 0],
            pads_end=[0, 0],
            kernel_shape=[2, 2],
        ).output(0)
        in_channels = out_channels

    return Model([node], [input_node], "cpu_vs_npu_cnn_benchmark")


def create_benchmark_model(
    np, Model, opset8, model_type, batch, features, layers, channels, height, width, dtype
):
    if model_type == "mlp":
        return create_mlp_model(np, Model, opset8, batch, features, layers, dtype)
    return create_cnn_model(np, Model, opset8, batch, channels, height, width, dtype)


def build_input_data(np, model_type, batch, features, channels, height, width, dtype):
    tensor_dtype = np.float16 if dtype == "fp16" else np.float32
    rng = np.random.default_rng(0)
    if model_type == "mlp":
        return rng.standard_normal((batch, features)).astype(tensor_dtype)
    return rng.standard_normal((batch, channels, height, width)).astype(tensor_dtype)


def run_sync_requests(compiled_model, input_name, input_data, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        compiled_model.infer_new_request({input_name: input_data})
    return time.perf_counter() - start


def run_async_requests(compiled_model, input_name, input_data, iterations, num_requests):
    infer_requests = [
        compiled_model.create_infer_request() for _ in range(num_requests)
    ]
    started = [False] * num_requests

    start = time.perf_counter()
    for index in range(iterations):
        request_index = index % num_requests
        if started[request_index]:
            infer_requests[request_index].wait()
        infer_requests[request_index].start_async({input_name: input_data})
        started[request_index] = True

    for request, was_started in zip(infer_requests, started):
        if was_started:
            request.wait()
    return time.perf_counter() - start


def benchmark_device(
    core,
    np,
    model,
    device,
    iterations,
    warmup,
    model_type,
    batch,
    features,
    channels,
    height,
    width,
    dtype,
    hint,
    num_requests,
    repeats,
):
    compile_config = {"PERFORMANCE_HINT": hint}
    compiled_model = core.compile_model(model, device, compile_config)
    input_port = compiled_model.input(0)
    input_name = input_port.get_any_name()
    input_data = build_input_data(
        np, model_type, batch, features, channels, height, width, dtype
    )

    if num_requests == 1:
        run_sync_requests(compiled_model, input_name, input_data, warmup)
    else:
        run_async_requests(
            compiled_model, input_name, input_data, warmup, num_requests
        )

    run_results = []
    for _ in range(repeats):
        if num_requests == 1:
            elapsed = run_sync_requests(compiled_model, input_name, input_data, iterations)
        else:
            elapsed = run_async_requests(
                compiled_model, input_name, input_data, iterations, num_requests
            )

        avg_latency_ms = (elapsed / iterations) * 1000.0
        throughput_fps = iterations / elapsed if elapsed > 0 else 0.0
        run_results.append(
            {
                "elapsed_s": elapsed,
                "avg_latency_ms": avg_latency_ms,
                "throughput_fps": throughput_fps,
            }
        )

    median_latency_ms = statistics.median(
        result["avg_latency_ms"] for result in run_results
    )
    median_throughput_fps = statistics.median(
        result["throughput_fps"] for result in run_results
    )
    median_elapsed_s = statistics.median(result["elapsed_s"] for result in run_results)

    return {
        "device": device,
        "iterations": iterations,
        "warmup": warmup,
        "elapsed_s": median_elapsed_s,
        "avg_latency_ms": median_latency_ms,
        "throughput_fps": median_throughput_fps,
        "hint": hint,
        "num_requests": num_requests,
        "repeats": repeats,
        "runs": run_results,
    }


def print_results(results):
    print_header("BENCHMARK RESULTS")
    for result in results:
        print("Device: {}".format(result["device"]))
        print("  Hint: {}".format(result["hint"]))
        print("  Num requests: {}".format(result["num_requests"]))
        print("  Repeats: {}".format(result["repeats"]))
        print("  Warmup: {}".format(result["warmup"]))
        print("  Iterations: {}".format(result["iterations"]))
        print("  Median total time: {:.4f} s".format(result["elapsed_s"]))
        print("  Median latency: {:.3f} ms".format(result["avg_latency_ms"]))
        print("  Median throughput: {:.2f} FPS".format(result["throughput_fps"]))
        for index, run_result in enumerate(result["runs"], start=1):
            print(
                "  Run {}: {:.3f} ms, {:.2f} FPS".format(
                    index, run_result["avg_latency_ms"], run_result["throughput_fps"]
                )
            )
        print()

    by_device = {result["device"]: result for result in results}
    if "CPU" in by_device and "NPU" in by_device:
        cpu = by_device["CPU"]
        npu = by_device["NPU"]

        latency_ratio = cpu["avg_latency_ms"] / npu["avg_latency_ms"]
        throughput_ratio = npu["throughput_fps"] / cpu["throughput_fps"]

        print("Comparison")
        print("  CPU latency / NPU latency: {:.2f}x".format(latency_ratio))
        print("  NPU throughput / CPU throughput: {:.2f}x".format(throughput_ratio))

        if latency_ratio > 1:
            print("  NPU is faster in average latency for this workload.")
        else:
            print("  CPU is faster in average latency for this workload.")


def build_summary(results):
    summary = {}
    by_device = {result["device"]: result for result in results}
    for left, right in [("CPU", "GPU"), ("CPU", "NPU"), ("GPU", "NPU")]:
        if left in by_device and right in by_device:
            left_result = by_device[left]
            right_result = by_device[right]
            summary["{}_vs_{}".format(left.lower(), right.lower())] = {
                "latency_ratio": left_result["avg_latency_ms"] / right_result["avg_latency_ms"],
                "throughput_ratio": right_result["throughput_fps"] / left_result["throughput_fps"],
            }
    return summary


def write_json_report(path, args, available_devices, results):
    config = {
        "model_type": args.model_type,
        "batch": args.batch,
        "features": args.features,
        "layers": args.layers,
        "channels": args.channels,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "hint": args.hint,
        "num_requests": args.num_requests,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
    }
    report = {
        "openvino_devices": sorted(available_devices),
        "config": config,
        "results": results,
        "summary": build_summary(results),
    }
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2)
    print("\nJSON report written to {}".format(path))


def print_subprocess_result(title, cmd):
    print("\n{}".format(title))
    result = run_command(cmd)
    if isinstance(result, Exception):
        print("   Failed to run {}: {}".format(cmd, result))
        return

    output = (result.stdout or "").strip()
    if output:
        for line in output.splitlines():
            print("   {}".format(line))
    else:
        print("   No output")


def print_device_properties(core, device, property_names):
    print("\n{} properties".format(device))
    for property_name in property_names:
        try:
            value = core.get_property(device, property_name)
            print("   {}: {}".format(property_name, value))
        except Exception as exc:
            print("   {}: unavailable ({})".format(property_name, exc))


def try_compile_probe(core, model, device, label):
    print("\nCompile probe: {} on {}".format(label, device))
    try:
        compiled_model = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
        print("   Compile: OK")
        input_port = compiled_model.input(0)
        shape = list(input_port.shape)
        print("   Input shape: {}".format(shape))
    except Exception as exc:
        print("   Compile: FAIL")
        print("   Error: {}".format(exc))


def run_probe(args, np, openvino_module, core, Model, opset8):
    print_header("NPU PROBE")
    print("OpenVINO version: {}".format(getattr(openvino_module, "__version__", "unknown")))
    print("Python executable: {}".format(sys.executable))
    print("Detected devices: {}".format(core.available_devices))

    print_subprocess_result(
        "ldconfig ze_loader",
        ["sh", "-lc", "ldconfig -p | grep ze_loader"],
    )
    print_subprocess_result(
        "Find ze libraries",
        ["sh", "-lc", "find /usr/lib /usr/local/lib -maxdepth 2 -name 'libze*' 2>/dev/null | sort"],
    )

    npu_properties = [
        "FULL_DEVICE_NAME",
        "SUPPORTED_PROPERTIES",
        "DEVICE_ARCHITECTURE",
        "DEVICE_GOPS",
        "OPTIMIZATION_CAPABILITIES",
        "RANGE_FOR_ASYNC_INFER_REQUESTS",
        "RANGE_FOR_STREAMS",
    ]
    print_device_properties(core, "NPU", npu_properties)

    if "GPU" in core.available_devices:
        gpu_properties = [
            "FULL_DEVICE_NAME",
            "SUPPORTED_PROPERTIES",
            "OPTIMIZATION_CAPABILITIES",
            "RANGE_FOR_ASYNC_INFER_REQUESTS",
            "RANGE_FOR_STREAMS",
        ]
        print_device_properties(core, "GPU", gpu_properties)

    probe_models = [
        (
            "mlp_tiny",
            create_mlp_model(np, Model, opset8, 1, 64, 1, args.dtype),
        ),
        (
            "mlp_small",
            create_mlp_model(np, Model, opset8, 1, 256, 2, args.dtype),
        ),
        (
            "cnn_tiny",
            create_cnn_model(np, Model, opset8, 1, 3, 32, 32, args.dtype),
        ),
        (
            "cnn_small",
            create_cnn_model(np, Model, opset8, 1, 3, 64, 64, args.dtype),
        ),
        (
            "cnn_default",
            create_cnn_model(np, Model, opset8, 1, 3, 224, 224, args.dtype),
        ),
    ]

    for label, model in probe_models:
        try_compile_probe(core, model, "NPU", label)
        if "GPU" in core.available_devices:
            try_compile_probe(core, model, "GPU", label)

    print_header("PROBE INTERPRETATION")
    print("1. If all probes fail, the issue is likely NPU runtime/driver mismatch.")
    print("2. If only larger probes fail, the runtime is partially working but limited.")
    print("3. If tiny probes pass and benchmark fails, the issue is likely graph support.")
    print("4. Compare this output before and after changing Level Zero or NPU runtime.")


def main():
    args = parse_args()
    np, openvino_module, Core, Model, opset8 = import_dependencies()

    core = Core()
    available_devices = set(core.available_devices)

    if not args.skip_diagnostics:
        print_diagnostics(core)

    if args.probe:
        if "NPU" not in available_devices:
            print_header("NPU PROBE FAILED")
            print("NPU is not available in OpenVINO.")
            sys.exit(1)
        run_probe(args, np, openvino_module, core, Model, opset8)
        return

    requested_devices = args.device or ["CPU", "NPU"]
    devices_to_test = [device for device in requested_devices if device in available_devices]

    if not devices_to_test:
        print_header("NO DEVICE AVAILABLE FOR BENCHMARK")
        print("Requested devices: {}".format(requested_devices))
        print("Available devices: {}".format(sorted(available_devices)))
        sys.exit(1)

    model = create_benchmark_model(
        np=np,
        Model=Model,
        opset8=opset8,
        model_type=args.model_type,
        batch=args.batch,
        features=args.features,
        layers=args.layers,
        channels=args.channels,
        height=args.height,
        width=args.width,
        dtype=args.dtype,
    )

    print_header("BENCHMARK CONFIG")
    print("Devices: {}".format(devices_to_test))
    print("Model type: {}".format(args.model_type))
    print("Batch size: {}".format(args.batch))
    if args.model_type == "mlp":
        print("Feature size: {}".format(args.features))
        print("Layers: {}".format(args.layers))
    else:
        print("Input shape: [{}, {}, {}, {}]".format(args.batch, args.channels, args.height, args.width))
    print("Warmup iterations: {}".format(args.warmup))
    print("Measured iterations: {}".format(args.iterations))
    print("Performance hint: {}".format(args.hint))
    print("Num requests: {}".format(args.num_requests))
    print("Repeats: {}".format(args.repeats))
    print("Tensor dtype: {}".format(args.dtype))

    results = []
    for device in devices_to_test:
        try:
            print("\nRunning benchmark on {}...".format(device))
            result = benchmark_device(
                core=core,
                np=np,
                model=model,
                device=device,
                iterations=args.iterations,
                warmup=args.warmup,
                model_type=args.model_type,
                batch=args.batch,
                features=args.features,
                channels=args.channels,
                height=args.height,
                width=args.width,
                dtype=args.dtype,
                hint=args.hint,
                num_requests=args.num_requests,
                repeats=args.repeats,
            )
            results.append(result)
        except Exception as exc:
            print("Failed on {}: {}".format(device, exc))

    if not results:
        print_header("BENCHMARK FAILED")
        print("No device completed successfully.")
        sys.exit(1)

    print_results(results)

    if args.json_out:
        write_json_report(args.json_out, args, available_devices, results)

    print_header("HOW TO INTERPRET")
    print("1. Median latency is better when lower.")
    print("2. Median throughput is better when higher.")
    print("3. THROUGHPUT plus multiple requests is more NPU-friendly than LATENCY mode.")
    print("4. For fair comparison, keep the same model, input shape, batch, and dtype.")
    print("5. If NPU still loses, increase --batch, image size, or use a real model.")


if __name__ == "__main__":
    main()
