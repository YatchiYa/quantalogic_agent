curl -X POST "http://localhost:8082/api/chrome/schedule" \
 -H "Content-Type: application/json" \
 -d '{
"name": "Daily Chrome Task",
"execution_config": {
"url": "https://example.com",
"headless": true,
"timeout_seconds": 60,
"additional_args": ["--no-sandbox", "--disable-gpu"]
},
"schedule_type": "time",
"times": [
{
"hour": 9,
"minute": 0,
"second": 0
},
{
"hour": 15,
"minute": 30,
"second": 0
}
],
"enabled": true
}'

curl -X POST "http://localhost:8082/api/chrome/instagram/schedule" \
 -H "Content-Type: application/json" \
 -d '{
"name": "Daily Instagram Content Generation",
"instagram_config": {
"content_context": "AI in studies law and lawfirms",
"num_images": 1,
"generate_images": true,
"analysis_model": "gemini/gemini-2.0-flash",
"content_model": "gemini/gemini-2.0-flash",
"image_model": "gemini/gemini-2.0-flash",
"image_generator": "stable_diffusion"
},
"schedule_type": "time",
"times": [
{
"hour": 10,
"minute": 0,
"second": 0
}
],
"enabled": true
}'

curl -X POST "http://localhost:8082/api/chrome/instagram/schedule/98da0ec5-171d-48fb-b2e8-2fb08925c496/execute"

curl -X GET "http://localhost:8082/api/chrome/instagram/schedules"
