# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # The shape of the library itself.
    class TestCirce < TestCase
        def test_version_is_defined
            assert_match(/\A\d+\.\d+/, Circe::VERSION)
        end

        def test_error_belongs_to_the_standard_family
            assert_operator Circe::Error, :<, StandardError
        end

        def test_analyze_returns_features_and_image
            result = circe.analyze(Image.rgb)

            assert_kind_of Array, result
            assert_equal 2, result.size
            assert_kind_of Array, result.first
        end

        def test_without_a_format_no_image_is_produced
            assert_nil circe.analyze(Image.rgb).last
            assert_nil circe.analyze(Image.rgb, nil).last
        end

        def test_a_featureless_image_yields_no_features
            assert_empty circe.analyze(Image.rgb).first
        end
    end
end
