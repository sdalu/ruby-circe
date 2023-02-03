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

    s.extensions  = [ 'ext/extconf.rb' ]
    s.files       = %w[ circe.gemspec ]        +
                    Dir['ext/**/*.{cpp,h,rb}'] +
                    Dir['lib/**/*.rb'        ] +
                    Dir['data/*.onnx'        ]
end
