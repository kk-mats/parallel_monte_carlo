#include <iostream>
#include <random>
#include <iomanip>
#include <numeric>
#include <chrono>
#include <omp.h>


class random_generator final
{
public:
	explicit random_generator()
		: mt(std::random_device()())
	{}

	auto yeild() noexcept
	{
		return std::generate_canonical<double, std::numeric_limits<double>::digits>(this->mt);
	}

private:
	std::mt19937_64 mt;
};

class monte_carlo final
{
public:
	explicit monte_carlo() noexcept = default;

	auto parallel_calc(const int n_samples) const noexcept
	{
		std::vector<int> sample_set(omp_get_max_threads(), 0);
		int rest = n_samples;
		const int granularity = n_samples / omp_get_max_threads() + 1;
		std::for_each(sample_set.begin(), sample_set.end(), [&](auto& v) mutable { v = std::min(rest, granularity); rest -= granularity; });

		int in = 0;
#pragma omp parallel for reduction(+:in)
		for (int i=0; i<sample_set.size(); ++i)
		{
			in += this->count(sample_set[i]);
		}
		return (in / double(n_samples)) * 4;
	}

	auto sequential_calc(const int n_samples) const noexcept
	{
		return (this->count(n_samples) / double(n_samples)) * 4;
	}

private:
	int count(const int n_samples) const noexcept
	{
		random_generator rg;
		std::vector<std::pair<double, double>> samples(n_samples);
		std::generate(samples.begin(), samples.end(), [&]() { return std::make_pair(rg.yeild(), rg.yeild()); });
		return std::count_if(samples.begin(), samples.end(), [](const auto& p) { return std::pow(p.first, 2) + std::pow(p.second, 2) <= 1; });
	}
};


int main()
{
	constexpr int n_samples = 1000;
	const auto pstart = std::chrono::system_clock::now();
	const auto pv = monte_carlo().parallel_calc(n_samples);
	const auto pend = std::chrono::system_clock::now();
	std::cout << "parallel:" << std::endl;
	std::cout << "\tresult: " << pv << std::endl;
	std::cout << "\ttime: " << std::chrono::duration_cast<std::chrono::milliseconds>(pend - pstart).count() << "ms" << std::endl;

	const auto sstart = std::chrono::system_clock::now();
	const auto sv = monte_carlo().sequential_calc(n_samples);
	const auto send = std::chrono::system_clock::now();

	std::cout << std::endl;
	std::cout << "sequential:" << std::endl;
	std::cout << "\tresult: " << sv << std::endl;
	std::cout << "\ttime: " << std::chrono::duration_cast<std::chrono::milliseconds>(send - sstart).count() << "ms" << std::endl;

}