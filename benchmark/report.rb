# frozen_string_literal: true

# Driven by run.rb, which has already loaded circe and timed doing so.
# FORMAT=tsv emits one "key<tab>value" line per figure, which is what
# compare.rb reads back.

module CirceBench
    # label => [ positional args, keyword args ] for Circe#analyze
    CASES = { "classify"   => [ [],        { classify: true } ],
              "face"       => [ [],        { face:     true } ],
              "both"       => [ [],        {}                ],
              "both + jpg" => [ [ :jpg ],  {}                ],
              "both + png" => [ [ :png ],  {}                ] }.freeze

    module_function

    def collect(circe, data)
        started = clock
        found   = circe.analyze(data).first
        cold    = (clock - started) * 1000
        rows    = CASES.map do |label, (args, opts)|
            [ label, measure(ROUNDS) { circe.analyze(data, *args, **opts) } ]
        end

        { load: LOAD_MS, cold: cold, rows: rows,
          found: found, rss: rss, bytes: data.bytesize }
    end

    def render_human(result)
        path, height, width = Circe::ONNX_YOLO
        puts "circe #{Circe::VERSION} -- #{File.basename(path)} " \
             "@ #{width}x#{height}, #{result[:bytes]} B image, " \
             "#{ROUNDS} rounds"
        puts
        printf("  %-18s %7.0f ms\n", "load",         result[:load])
        printf("  %-18s %7.0f ms\n", "cold analyze", result[:cold])
        puts
        render_rows(result)
    end

    def render_rows(result)
        printf("  %-18s %7s %7s %7s\n", "", "min", "avg", "max")
        result[:rows].each { report(it.first, it.last) }
        kinds = result[:found].map(&:first).tally
                              .map { |kind, n| "#{kind}: #{n}" }.join(", ")
        puts
        printf("  %-18s %7d    %s\n", "features", result[:found].size, kinds)
        printf("  %-18s %7d MB\n", "peak rss", result[:rss]) if result[:rss]
    end

    def render_tsv(result)
        path, height, width = Circe::ONNX_YOLO
        puts "model\t#{File.basename(path)}"
        puts "size\t#{width}x#{height}"
        puts "load\t#{result[:load].round}"
        puts "cold\t#{result[:cold].round}"
        result[:rows].each do |label, times|
            puts "#{label.delete(' ')}\t#{times.first.round}"
        end
        puts "features\t#{result[:found].size}"
        puts "rss\t#{result[:rss]}" if result[:rss]
    end

    def run
        result = collect(Circe.new, image)
        if ENV["FORMAT"] == "tsv" then render_tsv(result)
        else                           render_human(result)
        end
    end
end

CirceBench.run
