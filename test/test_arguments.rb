# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Argument handling of Circe#analyze. The image used to be read as a
    # byte string without any check, so a non-string segfaulted the process.
    class TestArguments < TestCase
        def test_image_must_be_a_string
            assert_raises(TypeError) { circe.analyze(42,  :jpg) }
            assert_raises(TypeError) { circe.analyze(nil, :jpg) }
            assert_raises(TypeError) { circe.analyze(:jpg)      }
        end

        def test_image_is_required
            assert_raises(ArgumentError) { circe.analyze }
        end

        def test_extra_positional_arguments_are_rejected
            assert_raises(ArgumentError) do
                circe.analyze(Image.rgb, :jpg, :surplus)
            end
        end

        def test_format_must_be_a_symbol
            assert_raises(TypeError) { circe.analyze(Image.rgb, "jpg") }
        end

        def test_format_must_be_one_we_can_encode
            error = assert_raises(ArgumentError) do
                circe.analyze(Image.rgb, :gif)
            end

            assert_match(/:png, :jpg or nil/, error.message)
        end

        def test_unknown_keywords_are_rejected
            assert_raises(ArgumentError) do
                circe.analyze(Image.rgb, bogus: true)
            end
        end

        # Arguments are validated before the image is decoded, so a bad
        # format is reported even when the image is nonsense too
        def test_arguments_are_checked_before_decoding
            assert_raises(ArgumentError) { circe.analyze("nonsense", :gif) }
        end
    end
end
