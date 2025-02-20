import os

api_key = os.getenv("OPENAI_API_KEY")
test_var = os.getenv("TEST_VAR")

if api_key:
    print("OPENAI_API_KEY:", api_key)
else:
    print("OPENAI_API_KEY not found.")

if test_var:
    print("TEST_VAR:", test_var)
else:
    print("TEST_VAR not found.")