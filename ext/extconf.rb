require 'mkmf'

$PKGCONFIG =
    case RbConfig::CONFIG['host_os']
    when /bsd/ then '/usr/local/bin/pkgconf'
    end


cflags, ldflags, libs = pkg_config('opencv4')

$LDFLAGS  += " #{ldflags} #{libs}"
$INCFLAGS += " #{cflags}"
$CXXFLAGS += "-std=c++17"

create_makefile("circe/core")
