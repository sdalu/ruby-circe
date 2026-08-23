# frozen_string_literal: true

require "bundler"
require "rake/testtask"
require "yard"

Bundler::GemHelper.install_tasks

YARD::Rake::YardocTask.new do |t|
    t.files         = [ 'lib/**/*.rb', 'ext/ucl.c' ]
    t.options       = [ '-m', 'markdown' ]
    t.stats_options = [ '--list-undoc' ]
end

EXT_DIR = "ext"
EXT_SO  = File.join(EXT_DIR, "core.so")
LIB_SO  = File.join("lib", "circe", "core.so")

desc "Build the extension and put it where the library expects it"
task :compile do
    Dir.chdir(EXT_DIR) do
        ruby "extconf.rb"
        sh "make"
    end
    mkdir_p File.dirname(LIB_SO)
    # Already in place when lib/circe/core.so is a symlink to the build
    cp EXT_SO, LIB_SO unless File.identical?(EXT_SO, LIB_SO)
end

desc "Remove the build products"
task :clean do
    Dir.chdir(EXT_DIR) { sh "make clean" if File.exist?("Makefile") }
    rm_f [ EXT_SO, LIB_SO, File.join(EXT_DIR, "Makefile") ]
end

Rake::TestTask.new do |t|
    t.libs    << "test"
    t.pattern =  "test/test_*.rb"
    t.warning =  false
end

task test: :compile

task default: :test
