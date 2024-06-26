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

// Sequential layer builder
class Sequential{
    public:
        Sequential(std::vector<std::shared_ptr<Layer>>&& layers)
        : m_layers(layers){}

        torch::Tensor operator()(const torch::Tensor& x, const bool isTraining){
            
            torch::Tensor out = x;
            for(size_t i=0; i<m_layers.size(); ++i){
                if(isTraining){
                    m_layers[i]->train();
                } else{
                    m_layers[i]->eval();
                }

                out = m_layers[i]->forward(out);
            }

            return out;
        }

        std::vector<torch::Tensor*> parameters(){
            std::vector<torch::Tensor*> params = {};
            for(size_t i=0; i<m_layers.size(); ++i){
                std::vector<torch::Tensor*> curr_layer_params = m_layers[i]->parameters();
                params.insert(params.end(), curr_layer_params.begin(), curr_layer_params.end());
            }

            return params;
        }

        std::vector<std::shared_ptr<Layer>> m_layers;
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

// Flatten Consecutive layer
// This is a modification to the regular flatten layer.
// As in the Wavenet implementation, `numConsecutiveElems` number of 
// consecutive tokens are grouped together to propagate 2D tensors (excl. batch) 
// to the next layers. With the linear layers, this can help form the recurrent tree-like structure.
class FlattenConsecutive : public Layer{

    public:
        FlattenConsecutive(int numConsecutiveElems, std::string layer_name)
         : m_numConsecutiveElems(numConsecutiveElems), m_layer_name(layer_name){}

        torch::Tensor forward(const torch::Tensor& x) override{
            int B = x.size(0);
            int T = x.size(1); // current context window length
            int C = x.size(2); // concatenated embedding dimension size

            auto out = x.view({B, T/m_numConsecutiveElems, C*m_numConsecutiveElems});
            int flattned_context_win_len = out.size(1);
            if(flattned_context_win_len==1){
                out = out.squeeze(/*dim=*/1);
            }

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

    private:
        int m_numConsecutiveElems;
        std::string m_layer_name;
        bool isTraining;
};