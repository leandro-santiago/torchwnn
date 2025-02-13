/*
 MIT License

 Copyright (c) 2025 Leandro Santiago de Araújo

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

*/

#include <torch/extension.h>

#include <vector>

using torch::Tensor;

// Code adapted from ULEEN repository: https://github.com/ZSusskind/ULEEN
// NOTE: This is just a utility function
// Python loop handling is slow, this function is a bottleneck, and
// profiling shows that H3 computation is a significant portion of training time

torch::Tensor h3_hash(
    Tensor & inp,
    Tensor & hash_vals
) {
    const auto device = inp.device();

    // Choose between hash values and 0 based on input bits
    // This is done using a tensor product, which, oddly, seems to be faster
    // than using a conditional lookup (e.g. torch.where)
    Tensor selected_entries = torch::einsum("hb,db->bdh", {hash_vals, inp});

    // Perform an XOR reduction along the input axis (b dimension)
    Tensor reduction_result = torch::zeros(
        {inp.size(0), hash_vals.size(0)},
        torch::dtype(torch::kInt64).device(device));
    for (long int i = 0; i < hash_vals.size(1); i++) {
        reduction_result.bitwise_xor_(selected_entries[i]); // In-place XOR
    }

    return reduction_result;
}


/* This fuction executes h3 hash function for multiple inputs. */

torch::Tensor h3_multi_hash(
    Tensor & inputs,
    Tensor & hash_vals
) {
    const auto device = inputs.device();

    // inputs: i x n x t, where i = # samples, n = # neurons, t = tuple size
    // hash_vals: h x t, where h = # hahes, n = # neurons, t = tuple size    
    Tensor selected_entries = torch::einsum("int,ht->inth", {inputs, hash_vals});

    // Perform an XOR reduction along the input axis (t dimension)
    Tensor reduction_result = torch::zeros(
        {inputs.size(0), inputs.size(1), hash_vals.size(0)},
        torch::dtype(torch::kInt64).device(device));

    for (long int i = 0; i < inputs.size(0); i++) {
        for (long int j = 0; j < inputs.size(1); j++) {
            for (long int k = 0; k < hash_vals.size(1); k++) {                
                reduction_result[i][j].bitwise_xor_(selected_entries[i][j][k]); // In-place XOR
            }
        }
    }

    return reduction_result;
}

void filter_multi_add(
    Tensor * filters,
    Tensor & data
) {    
    for (long int i = 0; i < data.size(0); i++) {
        filters->scatter_(1, data[i], 1, "add");
    }
}

torch::Tensor filter_multi_rank(
    Tensor & filters,
    Tensor & data,
    int bleach // 0 - without bleach, > 0 - bleach value
) {    
    const auto device = data.device();

    Tensor response = torch::zeros(
        {data.size(0)},
        torch::dtype(torch::kInt64).device(device));

    if (bleach > 0) {
        for (long int i = 0; i < data.size(0); i++) {
            Tensor selected_entries = torch::gather(filters, 1, data[i]);
            auto [values, indexes] = torch::min(selected_entries, 1);
            response[i] += (values > bleach).sum();            
        }
    } else {
        for (long int i = 0; i < data.size(0); i++) {
            Tensor selected_entries = torch::clamp(torch::gather(filters, 1, data[i]), 0, 1);
            
            for (long int j = 0; j < data.size(1); j++) {
                int and_value = 1;
                for (long int k = 0; k < data.size(2); k++) {
                    and_value = and_value & selected_entries[j][k].item<int>();
                }

                response[i] += and_value;
            }
        }
    }     

    return response;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("h3_hash", &h3_hash, "Compute H3 hash function");
    m.def("h3_multi_hash", &h3_multi_hash, "Compute H3 hash function for several inputs");
    m.def("filter_multi_add", &filter_multi_add, "Add value for multiple filters");
    m.def("filter_multi_rank", &filter_multi_rank, "Calculate rank for multiple filters");
}
