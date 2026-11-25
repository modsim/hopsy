#ifndef HOPSY_BENCHMARK_BATCH_MEANS_H
#define HOPSY_BENCHMARK_BATCH_MEANS_H

#include <limits>
#include <stdexcept>
#include <hopsy/linalg.h>

namespace hopsy::benchmark {
    // Combine equal-length independent traces without making batches cross
    // a chain or run boundary.
    class BatchMeans {
      public:
        BatchMeans(Index dimension, Index batch_size)
            : batch_size_(batch_size), marginal_(Vector<double>::Zero(dimension)),
              long_run_(Vector<double>::Zero(dimension)) {
            if (batch_size < 1)
                throw std::invalid_argument("ESS requires a positive batch size");
        }

        void add(const Matrix<double> &samples) {
            const Index length = samples.cols();
            const Index batches = length / batch_size_;
            if (batches < 2 || length % batch_size_ != 0 || samples.rows() != marginal_.size() ||
                (trace_length_ != 0 && trace_length_ != length))
                throw std::invalid_argument("ESS requires equal-length traces of whole batches");
            trace_length_ = length;
            sample_count_ += length;

            for (Index d = 0; d < samples.rows(); ++d) {
                const auto values = samples.row(d);
                const double mean = values.mean();
                marginal_[d] += (values.array() - mean).square().sum() / (length - 1);
                double squared_batch_errors = 0.;
                for (Index i = 0; i < batches; ++i) {
                    const double error = values.segment(i * batch_size_, batch_size_).mean() - mean;
                    squared_batch_errors += error * error;
                }
                long_run_[d] += batch_size_ * squared_batch_errors / (batches - 1);
            }
        }

        Index sample_count() const noexcept { return sample_count_; }

        Vector<double> ess() const {
            Vector<double> result(marginal_.size());
            for (Index d = 0; d < result.size(); ++d) {
                // With equal-length independent traces, use their average
                // marginal and average asymptotic variance for the pooled mean.
                result[d] = long_run_[d] > 0. && marginal_[d] > 0.
                                ? sample_count_ * marginal_[d] / long_run_[d]
                                : std::numeric_limits<double>::quiet_NaN();
            }
            return result;
        }

      private:
        Index batch_size_, trace_length_ = 0, sample_count_ = 0;
        Vector<double> marginal_, long_run_;
    };
} // namespace hopsy::benchmark

#endif
