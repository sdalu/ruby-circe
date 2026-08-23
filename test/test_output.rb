# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Encoding of the annotated image.
    class TestOutput < TestCase
        JPEG_MAGIC = "\xFF\xD8\xFF".b
        PNG_MAGIC  = "\x89PNG".b

        def test_jpg_produces_a_jpeg
            assert image(:jpg).start_with?(JPEG_MAGIC), "not a JPEG"
        end

        def test_png_produces_a_png
            assert image(:png).start_with?(PNG_MAGIC), "not a PNG"
        end

        def test_the_image_keeps_its_dimensions
            encoded = circe.analyze(Image.rgb(64, 48), :png).last

            assert_equal [64, 48], PNG.dimensions(encoded)
        end

        def test_debug_overlays_the_inference_time
            refute_equal image(:jpg, debug: false), image(:jpg, debug: true)
        end

        # A nil keyword hash used to leave the flags reading uninitialized
        # stack, so the default has to be pinned down explicitly
        def test_debug_defaults_to_false
            assert_equal image(:jpg, debug: false), image(:jpg)
        end

        def test_debug_does_not_disturb_the_features
            assert_equal features, features(debug: true)
        end
    end
end
