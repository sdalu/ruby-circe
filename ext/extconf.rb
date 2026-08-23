require 'mkmf'

$PKGCONFIG =
    case RbConfig::CONFIG['host_os']
    when /bsd/ then '/usr/local/bin/pkgconf'
    end

pkgconf = $PKGCONFIG || 'pkg-config'

# Only the modules circe actually uses. Note that mkmf's pkg_config()
# cannot be used to get them: besides returning the flags it also appends
# every opencv module to $libs, and linking all ~50 of them (345 shared
# objects once their own dependencies are pulled in) costs about ten
# seconds of run-time linking at every startup on a raspberry pi.
$INCFLAGS += " " + `#{pkgconf} --cflags        opencv4`.chomp
$LDFLAGS  += " " + `#{pkgconf} --libs-only-L   opencv4`.chomp
$libs      = " -lopencv_core -lopencv_imgproc -lopencv_imgcodecs" \
             " -lopencv_dnn -lopencv_objdetect"

$CXXFLAGS += " -std=c++17"

create_makefile("circe/core")
