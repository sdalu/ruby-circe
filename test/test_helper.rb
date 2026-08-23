# frozen_string_literal: true

require "digest"
require "fileutils"
require "minitest/autorun"
require "net/http"
require "uri"
require "zlib"

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))

begin
    require "circe"
rescue LoadError => e
    abort <<~MESSAGE
        Cannot load the circe extension: #{e.message}

        Build it first, from the top of the repository:
            rake compile
    MESSAGE
end

# Support code shared by the test suite.
module CirceTest
    FIXTURE_DIR = File.expand_path("fixtures", __dir__)

    # Raised from test blocks, to be told apart from a genuine failure
    Boom = Class.new(StandardError)
end

require_relative "support/png"
require_relative "support/sample"
require_relative "support/test_case"
