# frozen_string_literal: true

# Latency benchmark: startup, the cost of each detector, and what
# encoding an annotated image adds. Reported as min/avg/max over N
# rounds, the minimum being the one to compare -- it is the least
# polluted by whatever else the machine is doing.
#
#   rake bench
#   rake bench N=20 IMAGE=frame.jpg
#   rake bench MODEL=data/yolo11n.onnx SIZE=640x480
#
# SIZE is width x height, and has to be what the model was exported
# with. The model is selected before circe is required, as the networks
# are built at that point -- which is what the reported load time is.

require "digest"
require "fileutils"
require "net/http"
require "uri"

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))

module CirceBench
    ROUNDS = Integer(ENV.fetch("N", "10"))

    module_function

    def clock = Process.clock_gettime(Process::CLOCK_MONOTONIC)

    def rss
        size = `ps -o rss= -p #{Process.pid} 2>/dev/null`.to_i
        size.positive? ? size / 1024 : nil
    end

    # [ path, height, width ] when a model was named, nil otherwise
    def selected_model
        model = ENV["MODEL"]
        return nil if model.nil?

        width, height = ENV.fetch("SIZE", "640x480").split("x")
                           .map { Integer(it) }
        [ File.expand_path(model), height, width ]
    end

    def image
        path = ENV["IMAGE"]
        return File.binread(path) if path

        sample or abort("no image: pass IMAGE=, or allow the sample " \
                        "used by the test suite to be fetched")
    end

    # Falls back on the image the test suite uses, reusing its fetcher
    # rather than duplicating the pinned URL and digest. Only reached
    # when IMAGE= was not given, so the benchmark still runs on its own.
    def sample
        unless defined?(::CirceTest)
            Object.const_set(:CirceTest, Module.new)
            ::CirceTest.const_set(:FIXTURE_DIR,
                                  File.expand_path("../test/fixtures",
                                                   __dir__))
        end
        require_relative "../test/support/sample"
        ::CirceTest::Sample.data
    rescue LoadError
        nil
    end

    def measure(rounds)
        yield
        times = Array.new(rounds) do
            started = clock
            yield
            (clock - started) * 1000
        end
        [ times.min, times.sum / times.size, times.max ]
    end

    def report(label, times)
        printf("  %-18s %7.0f %7.0f %7.0f ms\n", label, *times)
    end
end

# Model selection has to happen before the extension is loaded
if (chosen = CirceBench.selected_model)
    class Circe; end
    Circe.const_set(:ONNX_YOLO,  chosen)
    Circe.const_set(:ONNX_YUNET,
                    [ File.expand_path("../data/" \
                        "face_detection_yunet_2023mar.onnx", __dir__) ])
    started = CirceBench.clock
    require "circe/version"
    require "circe/core"
else
    started = CirceBench.clock
    require "circe"
end
LOAD_MS = (CirceBench.clock - started) * 1000

require_relative "report"
