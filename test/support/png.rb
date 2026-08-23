# frozen_string_literal: true

module CirceTest
    # Minimal PNG writer. The pixel formats circe has to cope with (grey,
    # RGB, alpha, 16 bit) are generated rather than committed, so the suite
    # needs neither an image library nor a directory of binary fixtures.
    module PNG
        SIGNATURE = "\x89PNG\r\n\x1a\n".b
        GREY      = 0
        RGB       = 2
        RGBA      = 6

        module_function

        def build(width, height, type, depth, rows)
            size   = [width, height].pack("N2")
            header = size + [depth, type, 0, 0, 0].pack("C5")
            body   = Zlib::Deflate.deflate(rows.map { "\x00".b + it }.join)

            SIGNATURE + chunk("IHDR", header) +
                chunk("IDAT", body) + chunk("IEND", "".b)
        end

        def chunk(type, data)
            [data.bytesize].pack("N") + type + data +
                [Zlib.crc32(type + data)].pack("N")
        end

        # Width and height of an encoded PNG, straight out of its IHDR
        def dimensions(data) = data[16, 8].unpack("N2")
    end

    # The generated fixture images themselves.
    module Image
        module_function

        def rgb(width = 64, height = 48)
            rows = pack(width, height) { |x, y| [x % 256, y % 256, 128] }
            PNG.build(width, height, PNG::RGB, 8, rows)
        end

        def grey(width = 64, height = 48)
            rows = pack(width, height) { |x, y| [(x + y) % 256] }
            PNG.build(width, height, PNG::GREY, 8, rows)
        end

        def rgba(width = 64, height = 48)
            rows = pack(width, height) { |x, y| [x % 256, y % 256, 128, 200] }
            PNG.build(width, height, PNG::RGBA, 8, rows)
        end

        def grey16(width = 64, height = 48)
            rows = Array.new(height) { |y|
                Array.new(width) { |x| ((x + y) * 257) % 65_536 }.pack("n*")
            }
            PNG.build(width, height, PNG::GREY, 16, rows)
        end

        def tiny = PNG.build(1, 1, PNG::RGB, 8, [[255, 0, 0].pack("C*")])

        def pack(width, height)
            Array.new(height) { |y|
                Array.new(width) { |x| yield(x, y) }.flatten.pack("C*")
            }
        end
    end
end
