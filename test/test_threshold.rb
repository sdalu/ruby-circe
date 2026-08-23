# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Detection thresholds used to be compile time constants in yolo.h.
    class TestThreshold < TestCase
        def scores(**threshold)
            features(threshold: threshold)
                .reject { it.first == :face }
                .map    { it.last }
        end

        def faces(**threshold)
            features(threshold: threshold).count { it.first == :face }
        end

        def test_lowering_the_score_keeps_more_objects
            assert_operator scores(score: 0.35).size, :>, scores.size
        end

        def test_raising_the_score_keeps_fewer_objects
            assert_operator scores(score: 0.90).size, :<, scores.size
        end

        def test_the_score_is_a_floor
            scores(score: 0.80).each { assert_operator it, :>=, 0.80 }
        end

        def test_an_impossible_score_keeps_nothing
            assert_empty scores(score: 1.0)
        end

        def test_the_face_threshold_is_honoured
            refute_equal 0, faces
            assert_equal 0, faces(face: 0.999)
        end

        def test_defaults_are_unchanged_by_an_empty_hash
            assert_equal features, features(threshold: {})
        end

        def test_out_of_range_is_refused
            [-0.1, 1.1].each do |value|
                assert_raises(ArgumentError) { features(threshold: { score: value }) }
            end
        end

        # A misspelt key must not be silently ignored: the caller would
        # believe a threshold had been applied when it had not
        def test_unknown_keys_are_refused
            error = assert_raises(ArgumentError) do
                features(threshold: { scrore: 0.5 })
            end

            assert_match(/:confidence, :score, :nms/, error.message)
        end

        def test_threshold_must_be_a_hash
            assert_raises(TypeError) { features(threshold: 0.5) }
        end

        def test_values_must_be_numeric
            assert_raises(TypeError) { features(threshold: { nms: "half" }) }
        end
    end
end
