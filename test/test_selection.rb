# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # face: and classify: pick which detectors run. Selecting can be done
    # positively (run only what was named) or by exclusion (run everything
    # but what was refused); an explicit false used to be indistinguishable
    # from "not given" and quietly turned both detectors back on.
    class TestSelection < TestCase
        def test_nothing_given_runs_everything
            assert_equal %i[class face], kinds
        end

        def test_selecting_one_runs_only_that_one
            assert_equal %i[face],  kinds(face:     true)
            assert_equal %i[class], kinds(classify: true)
        end

        def test_refusing_one_still_runs_the_other
            assert_equal %i[class], kinds(face:     false)
            assert_equal %i[face],  kinds(classify: false)
        end

        def test_selecting_and_refusing_agree
            assert_equal %i[face],  kinds(face: true,  classify: false)
            assert_equal %i[class], kinds(face: false, classify: true )
        end

        def test_refusing_everything_runs_nothing
            assert_empty kinds(face: false, classify: false)
        end

        def test_selecting_everything_runs_everything
            assert_equal %i[class face], kinds(face: true, classify: true)
        end
    end
end
