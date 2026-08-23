# frozen_string_literal: true

module CirceTest
    # Real detections need a real photograph, and the suite would rather not
    # carry a binary for it. openCV's own sample data is used instead: it is
    # Apache-2.0, pinned to a release tag, checked against its digest, and
    # cached under test/fixtures once fetched.
    #
    # This particular image is convenient because it yields both a face and
    # several COCO classes, one of which is a person.
    module Sample
        URL = "https://raw.githubusercontent.com/opencv/opencv/" \
              "4.13.0/samples/data/messi5.jpg"
        SHA256 =
            "1d570e49654e84c7a943918537bd9e5e1ef82920152e147c834006e235be97c9"
        PATH = File.join(FIXTURE_DIR, "messi5.jpg")

        module_function

        # Bytes of the sample image, or nil when it is neither cached nor
        # reachable. Tests needing it skip themselves in that case, so that
        # an offline run reports honestly instead of failing.
        def data
            return @data if defined?(@data)

            @data = cached || downloaded
        end

        def cached
            return nil unless File.exist?(PATH)

            data = File.binread(PATH)
            data if Digest::SHA256.hexdigest(data) == SHA256
        end

        def downloaded
            data = fetch
            return nil unless data && Digest::SHA256.hexdigest(data) == SHA256

            FileUtils.mkdir_p(FIXTURE_DIR)
            File.binwrite(PATH, data)
            data
        end

        def fetch
            response = Net::HTTP.get_response(URI(URL))
            response.body if response.is_a?(Net::HTTPSuccess)
        rescue StandardError
            nil
        end
    end
end
