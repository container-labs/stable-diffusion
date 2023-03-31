require 'uri'
require 'oj'
require 'net/http'

Oj.default_options = {:mode => :compat }

def make_post_request(url, json_body)
  # Parse the URL and create the request
  uri = URI.parse(url)
  request = Net::HTTP::Post.new(uri)

  # Set the content type to JSON and add the JSON body to the request
  request.content_type = 'application/json'
  request.body = Oj.dump(json_body)
  puts(request.body)

  # Send the request
  Net::HTTP.start(uri.hostname, uri.port) do |http|
    http.request(request)
  end
end

require 'thread'

# Create a queue to hold the responses
response_queue = Queue.new

# Create a thread pool with 3 threads (1 per k8s pod)
threads = 3.times.map do
  Thread.new do
    # Dequeue a URL and JSON body from the queue
    url, json_body = response_queue.pop

    # Make the POST request and put the response into the queue
    puts "making request"
    response = make_post_request(url, json_body)
    puts response
    response_queue.push(response)
  end
end

# Enqueue the requests into the queue
100.times do |i|
  url = 'http://localhost:6000'
  json_body = {
    phrase: "highly detailed digital painting of beebz in space in the style of dopeaf",
    steps: 300,
    height: 768,
    width: 768,
    model: "/mnt/md-ml-public/training-job-1672631831/model"
  }
  response_queue.push([url, json_body])
end

# Wait for all threads to complete
threads.each(&:join)

# # Dequeue and print the responses
# 100.times do
#   response = response_queue.pop
#   puts response.body
# end
