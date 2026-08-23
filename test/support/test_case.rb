# frozen_string_literal: true

module CirceTest
    # Base class for the suite: hands every test a circe instance and quick
    # access to the sample photograph, skipping rather than failing when
    # that sample cannot be obtained.
    class TestCase < Minitest::Test
        def circe = @circe ||= Circe.new

        def sample
            data = Sample.data
            skip("sample image is neither cached nor reachable") unless data

            data
        end

        # Features of the sample image, memoized for the common no-option
        # case as each analysis costs a few hundred milliseconds
        def features(**opts)
            return @features ||= circe.analyze(sample).first if opts.empty?

            circe.analyze(sample, **opts).first
        end

        # Annotated rendering of the sample image
        def image(format = :jpg, **opts, &block)
            circe.analyze(sample, format, **opts, &block).last
        end

        def kinds(**opts) = features(**opts).map(&:first).uniq.sort
    end
end
