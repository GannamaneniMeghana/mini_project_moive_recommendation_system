from app import calculate_subscription_value, SERVICE_TITLES

print("Loaded SERVICE_TITLES keys:", list(SERVICE_TITLES.keys()))

# Test Case 1: Netflix and Prime exclusives
titles_1 = ["Stranger Things", "The Boys"]
print(f"\nTesting with: {titles_1}")
result_1 = calculate_subscription_value(titles_1)
print("Bundle:", [b['service'] for b in result_1['bundle']])
print("Total Cost:", result_1['total_cost'])

# Test Case 2: Overlapping or multiple
# Assuming "Inception" might be on multiple or just one.
# Check actual titles in CSV if possible, but basic logic test:
print("\nTesting with empty list:")
result_empty = calculate_subscription_value([])
print("Bundle:", result_empty['bundle'])

# Test Case 3: Uncoverable
titles_uncoverable = ["Some Fake Movie 123"]
print(f"\nTesting with uncoverable: {titles_uncoverable}")
result_uncoverable = calculate_subscription_value(titles_uncoverable)
print("Bundle:", result_uncoverable['bundle'])
print("Coverage:", result_uncoverable['coverage_pct'])
