# frozen_string_literal: true

# Driven by run.rb, which has already loaded circe and timed doing so.

module CirceBench
    module_function

    def header(data)
        path, height, width = Circe::ONNX_YOLO
        puts "circe #{Circe::VERSION} -- #{File.basename(path)} " \
             "@ #{width}x#{height}, #{data.bytesize} B image, " \
             "#{ROUNDS} rounds"
        puts
        printf("  %-18s %7.0f ms\n", "load", LOAD_MS)
    end

    def startup(circe, data)
        started = clock
        found   = circe.analyze(data).first
        printf("  %-18s %7.0f ms\n", "cold analyze", (clock - started) * 1000)
        puts
        found
    end

    def detectors(circe, data)
        printf("  %-18s %7s %7s %7s\n", "", "min", "avg", "max")
        { "classify"     => [ [], { classify: true } ],
          "face"         => [ [], { face:     true } ],
          "both"         => [ [], {} ],
          "both + jpg"   => [ [ :jpg ], {} ],
          "both + png"   => [ [ :png ], {} ] }.each do |label, (args, opts)|
            times = measure(ROUNDS) { circe.analyze(data, *args, **opts) }
            report(label, times)
        end
    end

    def footer(circe, data, found)
        kinds = found.map(&:first).tally
                     .map { |kind, n| "#{kind}: #{n}" }.join(", ")
        puts
        printf("  %-18s %7d    %s\n", "features", found.size, kinds)
        printf("  %-18s %7d MB\n", "peak rss", rss) if rss
        circe # keep it alive until the memory has been read
    end

    def run
        data  = image
        circe = Circe.new
        header(data)
        found = startup(circe, data)
        detectors(circe, data)
        footer(circe, data, found)
    end
end

CirceBench.run
