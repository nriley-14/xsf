#include "../testing_utils.h"
#include <xsf/stats.h>
#include <xsf/third_party/kokkos/mdspan.hpp>

namespace {

double ks_2samp_exact(long long int n1, long long int n2, double d, bool two_sided) {
    std::vector<double> work(std::min(n1, n2) + 1);
    std::mdspan work_span(work.data(), work.size());
    return xsf::ks_2samp_exact(n1, n2, d, two_sided, work_span);
}

} // namespace

TEST_CASE("ks_2samp exact test", "[ks_2samp_exact][xsf_tests]") {

    SECTION("ks_2samp exact scipy reference values") {
        // Reference values computed with SciPy 2.0.0.dev0 using:
        //
        // import math
        // from scipy.stats._stats_py import _attempt_exact_2kssamp
        //
        // n1, n2, d, alternative = 10, 12, 0.4, 'two-sided'
        // _attempt_exact_2kssamp(n1, n2, math.gcd(n1, n2), d, alternative)[2]
        //
        // two_sided=false corresponds to alternative='greater'.
        using test_case = std::tuple<long long int, long long int, double, bool, double, double>;
        auto [n1, n2, d, two_sided, expected, rtol] = GENERATE(
            // two-sided, n1 == n2
            test_case{5, 5, 0.6, true, 0.35714285714285715, 1e-15},
            test_case{1000, 1000, 0.05, true, 0.16411447756429137, 1e-15},
            // two-sided, n1 != n2
            test_case{10, 12, 0.4, true, 0.2829492488935213, 1e-15},
            test_case{12, 10, 0.4, true, 0.2829492488935213, 1e-15},
            test_case{9, 6, 0.5, true, 0.27352647352647347, 1e-15},
            test_case{37, 41, 0.3, true, 0.04522672719009359, 1e-15},
            test_case{100, 150, 0.13, true, 0.24772616277329038, 1e-15},
            test_case{3000, 20, 0.3, true, 0.04427986404884875, 1e-15},
            // one-sided, n1 == n2
            test_case{5, 5, 0.6, false, 0.17857142857142855, 1e-15},
            test_case{1000, 1000, 0.05, false, 0.08210211001282415, 1e-15},
            // one-sided, n1 != n2
            test_case{10, 12, 0.4, false, 0.14180092353466967, 1e-15},
            test_case{9, 6, 0.5, false, 0.13686313686313686, 1e-15},
            test_case{37, 41, 0.3, false, 0.02261351443319685, 1e-15},
            test_case{100, 150, 0.13, false, 0.1240833004333414, 1e-15},
            test_case{3000, 20, 0.3, false, 0.022140012803056455, 1e-15}
        );
        const double output = ks_2samp_exact(n1, n2, d, two_sided);
        const auto rel_error = xsf::extended_relative_error(output, expected);
        CAPTURE(n1, n2, d, two_sided, output, expected, rtol, rel_error);
        REQUIRE(rel_error <= rtol);
    }

    SECTION("ks_2samp exact d rounds to zero") {
        REQUIRE(ks_2samp_exact(10, 12, 0.0, true) == 1.0);
        REQUIRE(ks_2samp_exact(10, 12, 0.001, false) == 1.0);
    }

    SECTION("ks_2samp exact symmetry n1, n2") {
        REQUIRE(ks_2samp_exact(37, 41, 0.3, false) == ks_2samp_exact(41, 37, 0.3, false));
    }

    SECTION("ks_2samp exact unsuccessful") {
        // SciPy's _attempt_exact_2kssamp returns success=False for these, and
        // ks_2samp falls back to the asymptotic distribution.
        // Overflow in the outside method.
        REQUIRE(std::isnan(ks_2samp_exact(500, 700, 0.13, false)));
        // Result 1.0000000000000002 is outside [0, 1].
        REQUIRE(std::isnan(ks_2samp_exact(5, 5, 0.13, true)));
    }

    SECTION("ks_2samp exact invalid input") {
        REQUIRE(std::isnan(ks_2samp_exact(0, 5, 0.5, true)));
        REQUIRE(std::isnan(ks_2samp_exact(5, -1, 0.5, true)));
        REQUIRE(std::isnan(ks_2samp_exact(5, 6, std::numeric_limits<double>::quiet_NaN(), true)));
    }

    SECTION("ks_2samp exact work array too short") {
        // Needs min(n1, n2) + 1 = 6 entries.
        std::vector<double> work(5);
        std::mdspan work_span(work.data(), work.size());
        REQUIRE(std::isnan(xsf::ks_2samp_exact(5, 6, 0.5, true, work_span)));
    }
}
