#include <iostream>
#include <random>
#include <iomanip>
#include <numeric>
#include <mutex>
#include <omp.h>


class random_generator final
{
public:
	explicit random_generator()
		: mt(std::random_device()())
	{}

	double yeild() noexcept
	{
		return std::generate_canonical<double, std::numeric_limits<double>::digits>(this->mt);
	}

private:
	std::mt19937 mt;
};

class parallel_monte_carlo final
{
public:
	double calc(const int all, const int granularity)
	{
		std::vector<int> sample_set;

		int rest = all;
		for (; rest > granularity; rest -= granularity)
		{
			sample_set.push_back(granularity);
		}
		sample_set.push_back(rest);

		int in = 0;
		std::mutex iomutex;
#pragma omp parallel for reduction(+:in)
		for (int i=0; i<sample_set.size(); ++i)
		{
			in += this->count(sample_set[i], iomutex);
		}
		return (in / double(all)) * 4;
	}

private:
	int count(const int n_samples, std::mutex &iomutex)
	{
		random_generator rg;
		std::vector<std::pair<double, double>> samples(n_samples);
		std::generate(samples.begin(), samples.end(), [&]() { return std::make_pair(rg.yeild(), rg.yeild()); });
		int in=std::count_if(samples.begin(), samples.end(), [](const auto& p) { return std::pow(p.first, 2) + std::pow(p.second, 2) <= 1; });

		std::lock_guard<std::mutex> lock(iomutex);
		std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << "In thread id=" << omp_get_thread_num() << ", " << in << " of " << n_samples << " sample(s) hit inside the circle." << std::endl;
		return in;
	}
};

int main()
{
	std::cout << "result: " << parallel_monte_carlo().calc(100000000, 100000) << std::endl;

}