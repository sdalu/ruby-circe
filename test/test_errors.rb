# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Exceptions have to cross the boundary between ruby and C++ in both
    # directions, and neither language unwinds in a way the other follows.
    # A raise from the block longjmp()s over the C++ destructors; an openCV
    # throw reaching a ruby frame ends the process in std::terminate. Both
    # used to happen, so every case below is a regression test.
    class TestErrors < TestCase
        def test_a_raise_from_the_block_reaches_the_caller
            error = assert_raises(Boom) do
                circe.analyze(sample, :jpg) { |*| raise Boom, "from block" }
            end

            assert_equal "from block", error.message
        end

        def test_a_throw_from_the_block_reaches_its_catch
            caught = catch(:done) do
                circe.analyze(sample, :jpg) { |*| throw :done, 42 }
            end

            assert_equal 42, caught
        end

        def test_a_label_that_is_not_a_string_raises
            assert_raises(TypeError) do
                circe.analyze(sample, :jpg) { |*| { label: :symbol } }
            end
        end

        def test_a_label_with_a_null_byte_raises
            assert_raises(ArgumentError) do
                circe.analyze(sample, :jpg) { |*| { label: "a\0b" } }
            end
        end

        def test_a_colour_that_is_not_a_number_raises
            assert_raises(TypeError) do
                circe.analyze(sample, :jpg) { |*| { color: "red" } }
            end
        end

        def test_a_thickness_beyond_int_raises
            assert_raises(RangeError) do
                circe.analyze(sample, :jpg) { |*| { thickness: 2**40 } }
            end
        end

        # An openCV assertion, tripped from inside the block callback: the
        # hardest path, as it is a C++ throw nested inside rb_protect
        def test_an_opencv_failure_becomes_a_circe_error
            assert_raises(Circe::Error) do
                circe.analyze(sample, :jpg) { |*| { thickness: 100_000 } }
            end
        end

        def test_failures_leave_the_instance_usable
            3.times do
                assert_raises(Boom) do
                    circe.analyze(sample, :jpg) { |*| raise Boom }
                end
            end

            refute_empty circe.analyze(sample).first
        end
    end
end
