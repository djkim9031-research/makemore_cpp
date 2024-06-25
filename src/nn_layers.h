#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

// Layer base class
class Layer{
    public:
        virtual ~Layer() = default;
        virtual torch::Tensor forward(const torch::Tensor& x) = 0;
        virtual std::vector<torch::Tensor*> parameters() = 0;
        virtual std::string name() = 0;
        virtual void train() = 0;
        virtual void eval() = 0;
        virtual torch::Tensor weights() = 0;
        virtual torch::Tensor bias() = 0; 
};


// Linear layer
class Linear : public Layer{

    public:
        Linear(int fan_in, int fan_out, std::string layer_name, torch::Generator& gen, bool bias = true){
            m_weights = torch::randn({fan_in, fan_out}, gen)/std::sqrt(fan_in);
            m_weights.set_requires_grad(true);

            m_layer_name = layer_name;

            if(bias){
                m_bias = torch::zeros(fan_out);
                m_bias.set_requires_grad(true);
            } else{
                m_bias = torch::Tensor();
            }
        }

        torch::Tensor forward(const torch::Tensor& x) override{
            torch::Tensor out = torch::matmul(x, m_weights);
            if(m_bias.defined()){
                out += m_bias;
            }

            return out;
        }

        std::vector<torch::Tensor*> parameters() override{
            if(m_bias.defined()){
                return {&m_weights, &m_bias};
            }
            return {&m_weights};
        }

        std::string name() override{
            return m_layer_name;
        }

        void train() override{
            isTraining = true;
        }

        void eval() override{
            isTraining = false;
        }

        torch::Tensor weights(){
            return m_weights;
        }

        torch::Tensor bias(){
            return m_bias;
        }

    private:
        torch::Tensor m_weights;
        torch::Tensor m_bias;
        std::string m_layer_name;
        bool isTraining;
};


// Batchnorm 1D layer
class BatchNorm1D : public Layer{

    public:
        BatchNorm1D(int dim, std::string layer_name, double momentum = 0.1, double eps = 1e-5)
        : m_layer_name(layer_name), m_momentum(momentum), m_eps(eps), isTraining(true){

            // Trainable parameters
            m_gamma = torch::ones({1, dim});
            m_beta = torch::zeros({1, dim});
            m_gamma.set_requires_grad(true);
            m_beta.set_requires_grad(true);

            // Buffers
            m_running_mean = torch::zeros({1, dim});
            m_running_var = torch::ones({1, dim});
        }

        torch::Tensor forward(const torch::Tensor& x) override{
            torch::Tensor x_mean, x_var;
            if(isTraining){
                x_mean = x.mean(0, /*keepdim=*/true);
                x_var = x.var(0, /*unbiased=*/true, /*keepdim=*/true);
            } else{
                x_mean = m_running_mean;
                x_var = m_running_var;
            }

            torch::Tensor x_hat = (x - x_mean)/torch::sqrt(x_var + m_eps);
            torch::Tensor out = m_gamma*x_hat + m_beta;

            if(isTraining){
                torch::NoGradGuard no_grad;
                m_running_mean = (1 - m_momentum) * m_running_mean + m_momentum * x_mean;
                m_running_var = (1 - m_momentum) * m_running_var + m_momentum * x_var;
            }

            return out;
        }

        std::vector<torch::Tensor*> parameters() override{
            return {&m_gamma, &m_beta};
        }

        std::string name() override{
            return m_layer_name;
        }

        void train() override{
            isTraining = true;
        }

        void eval() override{
            isTraining = false;
        }

        torch::Tensor weights(){
            return torch::Tensor();
        }

        torch::Tensor bias(){
            return torch::Tensor();
        }

    private:
        double m_eps;
        double m_momentum;
        bool isTraining;
        std::string m_layer_name;
        torch::Tensor m_gamma;
        torch::Tensor m_beta;
        torch::Tensor m_running_mean;
        torch::Tensor m_running_var;
};


// Tanh activation layer
class TanhActivation : public Layer{

    public:
        TanhActivation(std::string layer_name)
        : isTraining(true), m_layer_name(layer_name) {}

        torch::Tensor forward(const torch::Tensor& x) override{
            auto out = torch::tanh(x); 
            output = out; // For tanh distribution visualization
            return out;
        }

        std::vector<torch::Tensor*> parameters() override{
            return {};
        }

        std::string name() override{
            return m_layer_name;
        }

        void train() override{
            isTraining = true;
        }

        void eval() override{
            isTraining = false;
        }

        torch::Tensor weights(){
            return torch::Tensor();
        }

        torch::Tensor bias(){
            return torch::Tensor();
        }

        torch::Tensor output;
    private:
        bool isTraining;
        std::string m_layer_name;

};


// Embedding layer
class Embedding : public Layer{

    public:
        Embedding(int vocab_size, int embedding_dims, std::string layer_name, torch::Generator& gen)
            : m_vocab_size(vocab_size), m_embedding_dims(embedding_dims), m_layer_name(layer_name){
            m_weights = torch::randn({vocab_size, embedding_dims}, gen);
            m_weights.set_requires_grad(true);
        }

        torch::Tensor forward(const torch::Tensor& x) override{
            torch::Tensor out = torch::nn::functional::one_hot(x, m_vocab_size).to(torch::kFloat32);
            return torch::matmul(out, m_weights);
        }

        std::vector<torch::Tensor*> parameters() override{
            return {&m_weights};
        }

        std::string name() override{
            return m_layer_name;
        }

        void train() override{
            isTraining = true;
        }

        void eval() override{
            isTraining = false;
        }

        torch::Tensor weights(){
            return m_weights;
        }

        torch::Tensor bias(){
            return torch::Tensor();
        }

    private:
        int m_vocab_size;
        int m_embedding_dims;
        torch::Tensor m_weights;
        std::string m_layer_name;
        bool isTraining;

};

// Flatten layer
class Flatten : public Layer{

    public:
        Flatten(std::string layer_name)
         : m_layer_name(layer_name){}

        torch::Tensor forward(const torch::Tensor& x) override{
            return x.view({x.size(0), -1});
        }

        std::vector<torch::Tensor*> parameters() override{
            return {};
        }

        std::string name() override{
            return m_layer_name;
        }

        void train() override{
            isTraining = true;
        }

        void eval() override{
            isTraining = false;
        }

        torch::Tensor weights(){
            return torch::Tensor();
        }

        torch::Tensor bias(){
            return torch::Tensor();
        }

    private:
        std::string m_layer_name;
        bool isTraining;

        

    

};