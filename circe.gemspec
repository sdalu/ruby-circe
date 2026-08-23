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

    # 3.4 for `it`, used by the test suite and the documented examples
    s.required_ruby_version = '>= 3.4'

    s.extensions  = [ 'ext/extconf.rb' ]

    # The models that are actually loaded. data/ also holds alternatives
    # kept in the repository for comparison, and each of those would add
    # tens of megabytes to the gem, so they are named rather than globbed.
    # A rename must break the build here instead of quietly shipping a gem
    # without its model.
    models = %w[ data/face_detection_yunet_2023mar.onnx
                 data/yolo11n.onnx ]
    missing = models.reject { |f| File.exist?(File.join(__dir__, f)) }
    raise "circe.gemspec: no such model: #{missing.join(', ')}" \
        unless missing.empty?

    # Ship what git tracks, minus those extra models. A Dir glob picks up
    # whatever happens to be lying around, which matters here: a couple of
    # stray models took this gem from 10 MB to 123 MB without a word.
    #
    # Outside a checkout -- built from an unpacked tarball, say -- fall
    # back on an explicit list rather than a glob, for the same reason.
    s.files =
        begin
            Dir.chdir(__dir__) do
                out = IO.popen([ 'git', 'ls-files', '-z' ],
                               err: File::NULL, &:read)
                raise 'not a checkout' unless $?&.success?

                out.split("\x0").reject { |f|
                    f.start_with?('.') ||
                        (f.start_with?('data/') && !models.include?(f))
                }
            end
        rescue StandardError, SystemCallError
            %w[ circe.gemspec README.md Rakefile ]               +
                models                                           +
                Dir['ext/**/*.{cpp,h,rb}']                       +
                Dir['lib/**/*.rb']                               +
                Dir['test/**/*.rb']                              +
                Dir['benchmark/**/*.rb']
        end
end
