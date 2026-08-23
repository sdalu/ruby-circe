require_relative 'lib/circe/version'

Gem::Specification.new do |s|
    s.name        = 'circe'
    s.version     = Circe::VERSION
    s.summary     = "Face and object recognition"
    s.description =  <<~EOF

      Face and object recognition

      EOF

    s.homepage    = 'https://github.com/sdalu/ruby-circe'
    s.license     = 'MIT'

    s.authors     = [ "Stéphane D'Alu"  ]
    s.email       = [ 'sdalu@sdalu.com' ]

    s.metadata    = {
        'source_code_uri' => s.homepage,
        'bug_tracker_uri' => "#{s.homepage}/issues",
    }

    s.required_ruby_version = '>= 3.0'

    s.extensions  = [ 'ext/extconf.rb' ]

    # Ship what git tracks, and nothing else. A Dir glob picks up whatever
    # happens to be lying around, which matters here because data/ collects
    # ONNX models weighing tens of megabytes each: a stray couple of them
    # took this gem from 10 MB to 123 MB without a word.
    #
    # Outside a checkout -- built from an unpacked tarball, say -- fall
    # back on an explicit list rather than a glob, for the same reason.
    s.files =
        begin
            Dir.chdir(__dir__) do
                out = IO.popen([ 'git', 'ls-files', '-z' ],
                               err: File::NULL, &:read)
                raise 'not a checkout' unless $?&.success?

                out.split("\x0").reject { |f| f.start_with?('.') }
            end
        rescue StandardError, SystemCallError
            %w[ circe.gemspec README.md Rakefile
                data/yolo11n.onnx
                data/face_detection_yunet_2023mar.onnx ]         +
                Dir['ext/**/*.{cpp,h,rb}']                       +
                Dir['lib/**/*.rb']                               +
                Dir['test/**/*.rb']                              +
                Dir['benchmark/**/*.rb']
        end
end
