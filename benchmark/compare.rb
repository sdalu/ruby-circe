# frozen_string_literal: true

# Compare several models, or one model at several input sizes, side by
# side. Each configuration runs in its own process: the network is built
# when the extension is required, so a single process can only ever
# measure one of them.
#
#   rake bench:compare
#   rake bench:compare MODELS="data/yolo11n.onnx@640x480 other.onnx@640x640"
#   rake bench:compare N=20 IMAGE=frame.jpg
#
# Sizes are width x height and must be what the model was exported with;
# a mismatch is reported against that row rather than stopping the run.

require "open3"
require "rbconfig"

module CirceCompare
    RUNNER  = File.expand_path("run.rb", __dir__)
    DEFAULT = File.expand_path("../data", __dir__)
    COLUMNS = %w[load cold classify face both both+jpg rss features].freeze

    module_function

    # "path@WxH" -> [ path, "WxH" ], the size defaulting to SIZE
    def parse(token)
        path, size = token.split("@")
        [ File.expand_path(path), size || ENV.fetch("SIZE", "640x480") ]
    end

    def configurations
        given = ENV["MODELS"]
        return given.split.map { parse(it) } if given

        Dir["#{DEFAULT}/*.onnx"].reject { it.include?("yunet") }.sort
                                .map { parse(it) }
    end

    # [ figures, nil ] on success, [ nil, why ] when the child died
    def measure(path, size)
        env = ENV.to_h.merge("MODEL" => path, "SIZE" => size,
                             "FORMAT" => "tsv")
        out, err, status = Open3.capture3(env, RbConfig.ruby, RUNNER)
        return [ nil, reason(err) ] unless status.success?

        [ out.lines.map { it.chomp.split("\t", 2) }.to_h, nil ]
    end

    # The interesting part of a child's death, not just "it failed": a
    # model may load and still be refused, and the message says why
    def reason(err)
        line = err.lines.find { it.match?(/Error/) } || err.lines.first
        text = line.to_s.sub(/\A.*?: /, "").chomp
                   .sub(/ \(\w[\w:]*\)\z/, "")
        # An openCV message leads with its build and the source file it
        # was raised from, neither of which says anything here
        text.sub(%r{.*?OpenCV\([^)]*\)\s+\S+\s+error:\s*}, "")[0, 68]
    end

    def row(path, size)
        result, why = measure(path, size)
        name        = "#{File.basename(path, '.onnx')} @#{size}"
        if result.nil?
            printf("  %-30s %s\n", name, why)
            return
        end
        values = COLUMNS.map { result.fetch(it, "-") }
        printf("  %-30s" + " %8s" * COLUMNS.size + "\n", name, *values)
    end

    def run
        configs = configurations
        abort("no model to compare") if configs.empty?

        puts "circe benchmark comparison -- min ms over " \
             "#{ENV.fetch('N', '10')} rounds"
        puts
        printf("  %-30s" + " %8s" * COLUMNS.size + "\n", "model", *COLUMNS)
        configs.each { row(*it) }
    end
end

CirceCompare.run
