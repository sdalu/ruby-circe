# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Slower checks on resource handling and re-entrancy.
    class TestRobustness < TestCase
        ROUNDS       = 30
        MAX_REGROWTH = 8 * 1024 # KiB

        # A raise from the block used to longjmp() over every C++
        # destructor, abandoning the decoded image on every single call.
        #
        # The first batch measures nothing useful: the allocator and the
        # inference engine claim tens of megabytes once and hold on to
        # them. What tells a leak apart from that warm-up is whether an
        # identical *second* batch grows all over again. Once the
        # destructors run it stays under a megabyte; while it leaked, the
        # second batch cost as much as the first.
        def test_block_exceptions_do_not_leak
            data = sample
            skip("resident size unavailable") if resident_size.nil?

            circe.analyze(data, :jpg) { |*| nil }
            warmup = batch_growth(data)

            assert_operator batch_growth(data), :<, MAX_REGROWTH,
                            "grew again after a #{warmup} KiB warm-up"
        end

        # Resident growth in KiB over one batch of failing calls
        def batch_growth(data)
            GC.start
            before = resident_size
            ROUNDS.times { failing_round(data) }
            GC.start

            resident_size - before
        end

        # Resolved before the threads start: a skip raised inside one would
        # surface as a thread crash rather than a skipped test
        def test_concurrent_calls_agree
            data    = sample
            threads = Array.new(4) do
                Thread.new { circe.analyze(data).first.size }
            end

            assert_equal 1, threads.map(&:value).uniq.size
        end

        def test_analyze_can_be_called_from_the_block
            data  = sample
            inner = nil
            circe.analyze(data, :jpg) do |*|
                inner ||= circe.analyze(data).first.size
                nil
            end

            assert_operator inner, :>, 0
        end

        def failing_round(data)
            circe.analyze(data, :jpg) { |*| raise Boom }
        rescue Boom
            nil
        end

        # Resident set size in KiB, or nil where ps cannot report it
        def resident_size
            size = `ps -o rss= -p #{Process.pid} 2>/dev/null`.to_i

            size.positive? ? size : nil
        end
    end
end
