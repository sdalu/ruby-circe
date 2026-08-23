# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Each feature is handed to the block, which says how it should be
    # drawn: as a label string, as [label, colour, thickness], or as a hash.
    class TestAnnotation < TestCase
        def test_the_block_is_called_once_per_feature
            count = 0
            found, = circe.analyze(sample, :jpg) do |*|
                count += 1
                nil
            end

            assert_equal found.size, count
        end

        def test_the_block_receives_the_whole_feature
            seen = []
            circe.analyze(sample, :jpg) do |type, box, *rest, confidence|
                seen << [type, box.size, rest.size, confidence.class]
                nil
            end

            assert_includes seen, [:face, 4, 1, Float]
        end

        def test_drawing_nothing_leaves_the_image_untouched
            untouched = image

            assert_equal untouched, image { |*| nil }
            assert_equal untouched, image { |*| { color: nil } }
        end

        def test_a_label_is_drawn
            refute_equal image, image { |*| "label" }
        end

        def test_a_bare_string_is_a_label
            assert_equal image { |*| { label: "l" } }, image { |*| "l" }
        end

        def test_array_and_hash_annotations_agree
            as_hash = image do |*|
                { label: "l", color: 0x00ff00, thickness: 3 }
            end

            assert_equal as_hash, image { |*| ["l", 0x00ff00, 3] }
        end

        def test_extra_draws_the_face_landmarks
            drawn = image(:jpg, face: true) { |*| { extra: true  } }

            refute_equal drawn, image(:jpg, face: true) { |*| { extra: false } }
        end

        # Known limitation: with no format there is no image to draw on, and
        # the block is not called at all
        def test_the_block_is_skipped_without_a_format
            called = false
            circe.analyze(sample) { |*| called = true }

            refute called
        end
    end
end
