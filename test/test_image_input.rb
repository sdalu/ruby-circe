# frozen_string_literal: true

require_relative "test_helper"

module CirceTest
    # Decoding used to abort the whole process rather than raise: openCV
    # threw a C++ exception straight through ruby on undecodable data, and
    # anything but three channels reached the network and failed an
    # assertion there. Greyscale photographs and PNGs with an alpha channel
    # are ordinary enough that both were easy to hit.
    class TestImageInput < TestCase
        def test_undecodable_data_raises
            assert_raises(Circe::Error) { circe.analyze("not an image") }
        end

        def test_empty_data_raises
            assert_raises(Circe::Error) { circe.analyze("") }
        end

        def test_truncated_image_raises
            assert_raises(Circe::Error) { circe.analyze(sample[0, 128]) }
        end

        def test_greyscale_is_accepted
            assert_empty circe.analyze(Image.grey).first
        end

        def test_alpha_is_accepted
            assert_empty circe.analyze(Image.rgba).first
        end

        def test_sixteen_bit_is_accepted
            assert_empty circe.analyze(Image.grey16).first
        end

        def test_single_pixel_is_accepted
            assert_empty circe.analyze(Image.tiny).first
        end

        # A degenerate input made the face detector emit a row of
        # uninitialised floats, reported as a face at INT_MIN. It only ever
        # showed up on the first detection of a process, so rather than
        # trying to reproduce that timing, assert the invariant it broke:
        # whatever comes out has to describe a region of the image.
        def test_features_always_describe_a_region_of_the_image
            cases = { grey: [Image.grey(64, 48), 64, 48],
                      rgb:  [Image.rgb(32, 32),  32, 32],
                      rgba: [Image.rgba(40, 30), 40, 30],
                      tiny: [Image.tiny,          1,  1] }

            cases.each do |name, (data, width, height)|
                circe.analyze(data).first.each do |_type, box, *|
                    assert_within_reach box, width, height, name
                end
            end
        end

        def assert_within_reach(box, width, height, format)
            x, y, w, h = box

            assert_operator w, :>, 0, "#{format}: empty width"
            assert_operator h, :>, 0, "#{format}: empty height"
            assert_operator x, :<, width,  "#{format}: starts past the right"
            assert_operator y, :<, height, "#{format}: starts below the bottom"
            assert_operator x + w, :>, 0,  "#{format}: ends left of the image"
            assert_operator y + h, :>, 0,  "#{format}: ends above the image"
        end

        # Whatever came in, what comes out is a plain three channel image
        def test_every_pixel_format_can_be_rendered
            %i[grey rgb rgba grey16 tiny].each do |format|
                data = circe.analyze(Image.send(format), :jpg).last

                refute_nil data, "#{format} produced no image"
            end
        end
    end
end
