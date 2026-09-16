import grader

def test_get_international_score():
	# Test case 1: No zones or tops
	assert grader.get_international_score([]) == 0.0

	assert grader.get_international_score([(1, None)]) == 10.0
	assert grader.get_international_score([(2, None)]) == 10.0-0.1
	assert grader.get_international_score([(3, None)]) == 10.0-0.2
	assert grader.get_international_score([(7, None)]) == 10.0-0.6

	assert grader.get_international_score([(1, 1)]) == 25.0
	assert grader.get_international_score([(1, 2)]) == 25.0-0.1
	assert grader.get_international_score([(2, 2)]) == 25.0-0.1
	assert grader.get_international_score([(1, 3)]) == 25.0-0.2
	assert grader.get_international_score([(2, 4)]) == 25.0-0.3
	assert grader.get_international_score([(1, 7)]) == 25.0-0.6
	assert grader.get_international_score([(7, 7)]) == 25.0-0.6

	# these cases shouldn't happen since the number of attempts for tops should always be greater than or equal to the number of attempts for zones, but we can still test them
	assert grader.get_international_score([(7, 2)]) == 25.0-0.1
	assert grader.get_international_score([(7, 2)]) == 25.0-0.1
	assert grader.get_international_score([(7, 3)]) == 25.0-0.2
	assert grader.get_international_score([(7, 4)]) == 25.0-0.3
	assert grader.get_international_score([(7, 7)]) == 25.0-0.6
	assert grader.get_international_score([(7, 7)]) == 25.0-0.6

	assert grader.get_international_score([(1, None), (1, None)]) == 20.0
	assert grader.get_international_score([(1, None), (2, None)]) == 20.0-0.1
	assert grader.get_international_score([(1, 1), (1, 1)]) == 50.0
	assert grader.get_international_score([(1, 1), (1, 2)]) == 50.0-0.1

	assert grader.get_international_score([(1, None), (1, 1), (1, 2), (7, 3)]) == 10.0 + 25.0 + (25.0-0.1) + (25.0-0.2)

test_get_international_score()
print("All tests passed for get_international_score()")