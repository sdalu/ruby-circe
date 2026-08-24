# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # The face detector runs at the size of the image it is handed, so
    # unlike a yolo pass its cost follows the source resolution. Beyond
    # face_input the image is shrunk for it and the coordinates scaled
    # back, which is the part that can be subtly wrong.
    class TestFaceInput < TestCase
        def faces(**opts) = features(face: true, **opts).select { it.first == :face }

        def box(**opts) = faces(**opts).first&.at(1)

        def test_an_image_under_the_cap_is_untouched
            # the sample is 548 px wide, below the default
            assert_equal faces(face_input: nil), faces
        end

        def test_zero_and_nil_both_mean_the_full_image
            assert_equal faces(face_input: nil), faces(face_input: 0)
        end

        def test_a_face_survives_being_shrunk_for_detection
            refute_nil box(face_input: 200)
        end

        # The coordinates must come back in the source image, not in the
        # shrunk one, so they have to land near the uncapped answer
        def test_coordinates_are_scaled_back
            reference = box(face_input: nil)
            scaled    = box(face_input: 300)

            scaled.zip(reference).each do |got, want|
                assert_in_delta want, got, 8
            end
        end

        def test_landmarks_are_scaled_back_too
            reference = faces(face_input: nil).first[2]
            scaled    = faces(face_input: 300).first[2]

            assert_equal reference.size, scaled.size
            reference.zip(scaled).each do |want, got|
                want.zip(got).each { |a, b| assert_in_delta a, b, 8 }
            end
        end

        def test_classification_is_unaffected
            classes = ->(**o) { features(**o).count { it.first == :class } }

            assert_equal classes.(), classes.(face_input: 128)
        end

        def test_a_negative_size_is_refused
            assert_raises(ArgumentError) { features(face_input: -1) }
        end

        def test_a_size_that_is_not_a_number_is_refused
            assert_raises(TypeError) { features(face_input: "640") }
        end
    end
end
