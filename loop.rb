require 'faraday'
require 'json'
require 'benchmark'

# all the things that matter arent there
# with what product

url = 'https://stable.gcp-gcp-gcp.com/'
for i in 1..5 do
  puts "started #{i}"
  time = Benchmark.measure {
    response = Faraday.get(url, {phrase: "Elon Musk riding a rocket", steps: 100})
    File.open("image-#{i}.png", 'wb') { |fp| fp.write(response.body) }
  }
  puts "finished #{i} in #{time.real.truncate(2)} seconds"
end

