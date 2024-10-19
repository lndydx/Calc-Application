import numpy as np
import matplotlib.pyplot as plt

P0 = int(input('P0 : ')) 
Px = int(input('Px : '))  
t0 = int(input('t0 : '))  
tx = int(input('tx : '))  

# Growth constant 
k = np.log(Px / P0) / (tx - t0) 

# Function to calculate population at a given year => P(t) = Po e^(kt)
def population(t, P0, k):
    return P0 * np.exp(k * (t - t0))

# Input years for which to calculate population
years_input = input('Input year (seperate with coma) : ')
years = [int(year) for year in years_input.split(',')]

populations = [population(year, P0, k) for year in years]

for year, pop in zip(years, populations):
    print(f"Population at {year}: {int(pop)} people")

years_range = np.arange(t0, max(years) + 1, 1)
pop_values = population(years_range, P0, k)

# Plot the exponential growth graph with prediction points
plt.figure(figsize=(8, 6))
plt.plot(years_range, pop_values, label='Exponential Population')
plt.scatter(years, populations, color='red', label='Prediction Point', zorder=5)
plt.title('population growth graph')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
 
# P0 = Initial Population
# Px = Population at x year
# t0 = Initial year
# tx = Year for Px
# k = Growth rate constant