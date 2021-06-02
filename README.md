# policy_iteration_car_rental_variant2

Implements a solution to the modified [Jack's Car Rental](http://www.incompleteideas.net/sutton/book/first/4/node4.html) problem in Sutton and Barto - 'An Introduction to Reinforcement Learning',
problem 4.5, and we believe it gives the correct results. 

We previously implemented the original variant of the problem [here](https://github.com/lucaslingle/policy_iteration_car_rental). 

### Figures

Here are the final value estimates after four policy improvement iterations:
![value estimates](assets/value_estimates.png)

Here is the final policy produced after four policy improvement iterations:
![policy](assets/policy.png)

In the figures above, the value estimates range from 429 to 603 and the policy ranges from -5 to 5.

### Interpretation

The yellow spike on the left-hand side, jutting towards the center of the grid, has a particularly interesting interpretation. 
It says that if we have 15 cars at store 0, we should always move 5 of them to store 1, if store 1 has between 0 and 5 cars.
This makes sense, since this variant of the problem penalizes having more than 10 cars overnight at any store; moving 5 cars to store 1 allows the policy to avoid a penalty.
Likewise, we can see that for 14 cars at store 0, we should move at least 4 of them to store 1, if store 1 has between 0 and 6 cars.

The vertical line in the middle, towards the bottom half of the grid, says to move zero cars from store 0 to store 1, if there are 12 or more cars at store 0, and 10 cars at store 1. 
Although moving one car would be the standard choice here (judging by the surrounding green basin), in this precise circumstance, 
moving one car at the end of the day would lead to penalties for both stores, because more than 10 cars would be stored overnight for both stores. 
Our implementation assumes the penalties are paid separately by both stores to separate overflow lots, so this makes sense.

However, if there are 11 cars at store 0, and 10 at store 1, the policy indicates to move one car to store 1. 
This makes sense since the surrounding area is also green, and moving one car would likely be optimal here if not for the penalty. 
Since exactly one penalty would be paid either way, with 11/10 and 10/11, the policy proceeds in moving one car.

The blue bar immediately to the right of the vertical bar discussed before, also has a partial interpretation. It says to move -1 cars from store 0 to store 1. 
In other words, transfer a car from store 1, with 11 cars, to store 0, if store 0 has between 13 and 19 cars. 
This allows store 0 to avoid a penalty for overnight storage of its cars, which would otherwise exceed 10.

The square below the blue line says to move one car from store 0 to store 1. This is quite a jump from the blue line, which says to do the exact opposite.
It can be explained by the fact that no more than 20 cars can be at a store, and hence transferring a car from store 1 to store 0 is not a valid action in this case, 
and hence the policy must choose between the other valid options instead. This explains the discontinuity. 
