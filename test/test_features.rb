# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # The feature tuples as documented in the README:
    #   :class, box, name,     confidence
    #   :face,  box, landmark, confidence
    class TestFeatures < TestCase
        def classes = features.select { it.first == :class }
        def faces   = features.select { it.first == :face  }

        def test_both_kinds_are_detected
            refute_empty classes
            refute_empty faces
        end

        def test_a_person_is_recognised
            assert_includes classes.map { it[2] }, "person"
        end

        # The names must be the modern COCO ones the models were trained
        # with, not the older VOC spellings circe used to carry: a filter
        # written against the model's own vocabulary has to match
        def test_class_names_use_the_modern_coco_vocabulary
            %w[motorcycle airplane couch tv].each do |name|
                assert_includes Circe::CLASSES, name
            end
            %w[motorbike aeroplane sofa tvmonitor].each do |name|
                refute_includes Circe::CLASSES, name
            end
        end

        def test_the_vocabulary_is_the_full_coco_set
            assert_equal 80, Circe::CLASSES.size
            assert_equal "person", Circe::CLASSES.first
            assert Circe::CLASSES.frozen?, "CLASSES should be frozen"
        end

        # Whatever is reported has to come from that vocabulary
        def test_detected_names_belong_to_the_vocabulary
            classes.each { assert_includes Circe::CLASSES, it[2] }
        end

        def test_every_feature_is_a_quadruple
            features.each { assert_equal 4, it.size }
        end

        def test_class_features_carry_a_name
            _type, box, name, confidence = classes.first

            assert_box box
            assert_kind_of String, name
            refute_empty name
            assert_confidence confidence
        end

        def test_face_features_carry_five_landmarks
            _type, box, landmark, confidence = faces.first

            assert_box box
            assert_equal 5, landmark.size
            landmark.each { assert_box(it, size: 2) }
            assert_confidence confidence
        end

        def test_boxes_have_a_positive_extent
            features.each do |_type, box, *|
                assert_operator box[2], :>, 0
                assert_operator box[3], :>, 0
            end
        end

        # The README lists the right eye first, then the left: on a frontal
        # face the right eye is the one at the smaller x
        def test_landmarks_start_with_the_right_eye
            right_eye, left_eye, = faces.first[2]

            assert_operator right_eye.first, :<, left_eye.first
        end

        def assert_box(box, size: 4)
            assert_kind_of Array, box
            assert_equal size, box.size
            box.each { assert_kind_of Integer, it }
        end

        def assert_confidence(value)
            assert_kind_of Float, value
            assert_includes 0.0..1.0, value
        end
    end
end
