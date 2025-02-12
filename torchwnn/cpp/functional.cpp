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
using namespace torch::indexing;

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

/* This fuction executes h3 hash function considering one integer input. */

torch::Tensor h3_hash_int(
    Tensor & inputs,
    Tensor & hash_vals
) {
    const auto device = inputs.device();

    Tensor binary_inputs = torch::empty(
        {inputs.size(0), hash_vals.size(1)},
        torch::dtype(torch::kUInt8).device(device));
    unsigned int val;
    unsigned int bit_position;
    
    for (long int i = 0; i < inputs.size(0); i++) {
        val = inputs[i].item<int>();
        bit_position = hash_vals.size(1) - 1;
          
        for (long int j = 0; j < hash_vals.size(1); j++) {
            binary_inputs[i][j] = (val >> bit_position) & 1;
            bit_position--;                        
        }
    }

    Tensor selected_entries = torch::einsum("hb,db->bdh", {hash_vals, binary_inputs});

    // Perform an XOR reduction along the input axis (b dimension)
    Tensor reduction_result = torch::zeros(
        {inputs.size(0), hash_vals.size(0)},
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

/* This fuction executes h3 hash function for multiple inputs considering integers inputs. */

torch::Tensor h3_multi_hash_int(
    Tensor & inputs,
    Tensor & hash_vals
) {
    const auto device = inputs.device();

    Tensor binary_inputs = torch::empty(
        {inputs.size(0), inputs.size(1), hash_vals.size(1)},
        torch::dtype(torch::kUInt8).device(device));
    unsigned int val;
    unsigned int bit_position;
    
    for (long int i = 0; i < inputs.size(0); i++) {
        for (long int j = 0; j < inputs.size(1); j++) {
            val = inputs[i][j].item<int>();
            bit_position = hash_vals.size(1) - 1;
            
            for (long int k = 0; k < hash_vals.size(1); k++) {  
                binary_inputs[i][j][k] = (val >> bit_position) & 1;
                bit_position--;            
            }            
        }
    }

    // inputs: i x n x t, where i = # samples, n = # neurons, t = tuple size
    // hash_vals: h x t, where h = # hahes, n = # neurons, t = tuple size    
    Tensor selected_entries = torch::einsum("int,ht->inth", {binary_inputs, hash_vals});

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

torch::Tensor sketch_multi_rank(
    Tensor & filters,
    Tensor & data,
    int bleach // 0 - without bleach, > 0 - bleach value
) {    
    const auto device = data.device();

    Tensor response = torch::zeros(
        {data.size(0)},
        torch::dtype(torch::kInt64).device(device));
    
    for (long int i = 0; i < data.size(0); i++) {
        Tensor selected_entries = torch::gather(filters, 1, data[i]);
        selected_entries.transpose_(1, 0);
        auto [values, indexes] = torch::min(selected_entries, 1);
        response[i] += (values > bleach).sum();
    }

    return response;
}

void cuckoo_filter_multi_add(
    Tensor & data, // data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint]  
    Tensor * filters, // filters: (n_neurons x bucket_size x filter_size)
    Tensor * bucket_indexes, // bucket_indexes: (n_neurons x filter_size)
    Tensor & fingerprint_matrix, // fingerprint_matrix: (1 x fingerprint_size)
    int maxKick    
) {   

    int i1, i2, fingerprint, fingerprint_aux, bucket_index1, bucket_index2;
    int bucket_size = filters->size(1);
    
    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            i1 = data[i][j][0].item<int>();
            i2 = data[i][j][1].item<int>();
            fingerprint = data[i][j][2].item<int>();
            bucket_index1 = bucket_indexes[0][j][i1].item<int>();
            bucket_index2 = bucket_indexes[0][j][i2].item<int>();
            
            if (bucket_index1 < bucket_size) {
                int selected = torch::isin(filters->index({j, Slice(None, None), i1}), fingerprint).any().item<int>();
                
                if (selected == 0) {
                    filters[0][j][bucket_index1][i1] = fingerprint;
                    bucket_indexes[0][j][i1] = bucket_index1 + 1;
                }
            } else if (bucket_index2 < bucket_size) {
                int selected = torch::isin(filters->index({j, Slice(None, None), i2}), fingerprint).any().item<int>();
                
                if (selected == 0) {
                    filters[0][j][bucket_index2][i2] = fingerprint;
                    bucket_indexes[0][j][i2] = bucket_index2 + 1;
                }                
            } else {
                int index = torch::randperm(2)[0].item<int>();
                index = data[i][j][index].item<int>();

                for (int k = 0; k < maxKick; k++) {
                    int bucket_i = torch::randperm(bucket_size)[0].item<int>();
                    // Swap fingerprint from bucket[index][bucket_i]
                    fingerprint_aux = filters[0][j][bucket_i][index].item<int>();
                    filters[0][j][bucket_i][index] = fingerprint;
                    fingerprint = fingerprint_aux;

                    // Calculate i = j xor hash(fingerprint), where i or j = i1 or i2
                    Tensor fingerprint_tensor = torch::tensor({fingerprint});
                    Tensor hash_results = h3_hash_int(fingerprint_tensor, fingerprint_matrix);                    
                    index = index ^ hash_results[0][0].item<int>();
                    bucket_index1 = bucket_indexes[0][j][index].item<int>();    

                    // If bucket has an empty entry, then fingerprint is inserted    
                    if (bucket_index1 < bucket_size) {
                        filters[0][j][bucket_index1][index] = fingerprint;
                        bucket_indexes[0][j][index] = bucket_index1 + 1;
                        break;
                    }
                }
            }            

        }
    }
}

void cuckoo_filter_multi_add_bleach(
    Tensor & data, // data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint]  
    Tensor * filters, // filters: (n_neurons x bucket_size x filter_size)
    Tensor * bucket_indexes, // bucket_indexes: (n_neurons x filter_size)
    Tensor & fingerprint_matrix, // fingerprint_matrix: (1 x fingerprint_size)
    Tensor * counters, // counters: (n_neurons x filter_size)
    int maxKick    
) {   

    int i1, i2, fingerprint, fingerprint_aux, bucket_index1, bucket_index2;
    int bucket_size = filters->size(1);
    
    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            i1 = data[i][j][0].item<int>();
            i2 = data[i][j][1].item<int>();
            fingerprint = data[i][j][2].item<int>();
            bucket_index1 = bucket_indexes[0][j][i1].item<int>();
            bucket_index2 = bucket_indexes[0][j][i2].item<int>();
            
            if (bucket_index1 < bucket_size) {
                int selected = torch::isin(filters->index({j, Slice(None, None), i1}), fingerprint).any().item<int>();
                
                if (selected == 0) {
                    filters[0][j][bucket_index1][i1] = fingerprint;
                    bucket_indexes[0][j][i1] = bucket_index1 + 1;
                } 
                counters[0][j][i1] = counters[0][j][i1].item<int>() + 1;
            } else if (bucket_index2 < bucket_size) {
                int selected = torch::isin(filters->index({j, Slice(None, None), i2}), fingerprint).any().item<int>();
                
                if (selected == 0) {
                    filters[0][j][bucket_index2][i2] = fingerprint;
                    bucket_indexes[0][j][i2] = bucket_index2 + 1;
                }                
                counters[0][j][i2] = counters[0][j][i2].item<int>() + 1;
            } else {
                int index = torch::randperm(2)[0].item<int>();
                index = data[i][j][index].item<int>();
                
                for (int k = 0; k < maxKick; k++) {
                    int bucket_i = torch::randperm(bucket_size)[0].item<int>();
                    // Swap fingerprint from bucket[index][bucket_i]
                    fingerprint_aux = filters[0][j][bucket_i][index].item<int>();
                    filters[0][j][bucket_i][index] = fingerprint;                    
                    fingerprint = fingerprint_aux;

                    // Calculate i = j xor hash(fingerprint), where i or j = i1 or i2
                    Tensor fingerprint_tensor = torch::tensor({fingerprint});
                    Tensor hash_results = h3_hash_int(fingerprint_tensor, fingerprint_matrix);                    
                    index = index ^ hash_results[0][0].item<int>();
                    bucket_index1 = bucket_indexes[0][j][index].item<int>();    

                    // If bucket has an empty entry, then fingerprint is inserted    
                    if (bucket_index1 < bucket_size) {
                        filters[0][j][bucket_index1][index] = fingerprint;
                        bucket_indexes[0][j][index] = bucket_index1 + 1;
                        counters[0][j][index] = counters[0][j][index].item<int>() + 1;                        
                        break;
                    }
                }
            }            

        }
    }
}

torch::Tensor cuckoo_filter_multi_rank(
    Tensor & filters, // filters: (n_neurons x bucket_size x filter_size)
    Tensor & data // data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint]  
) {    
    const auto device = data.device();

    Tensor response = torch::zeros(
        {data.size(0)},
        torch::dtype(torch::kInt64).device(device));
    
    int i1, i2, fingerprint, selected1, selected2;

    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            i1 = data[i][j][0].item<int>();
            i2 = data[i][j][1].item<int>();
            fingerprint = data[i][j][2].item<int>();
            
            selected1 = torch::isin(filters.index({j, Slice(None, None), i1}), fingerprint).any().item<int>();
            selected2 = torch::isin(filters.index({j, Slice(None, None), i2}), fingerprint).any().item<int>();

            response[i] += (selected1 | selected2);
        }
    }

    return response;
}

torch::Tensor cuckoo_filter_multi_rank_bleach(
    Tensor & filters, // filters: (n_neurons x bucket_size x filter_size)
    Tensor & counters, // counters: (n_neurons x filter_size)
    Tensor & data, // data: (n_samples x n_filters x 3), where last dimension has [i1, i2, fingerprint] 
    int bleach // 0 - without bleach, > 0 - bleach value
) {    
    const auto device = data.device();

    Tensor response = torch::zeros(
        {data.size(0)},
        torch::dtype(torch::kInt64).device(device));
    
    int i1, i2, fingerprint, selected1, selected2;

    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            i1 = data[i][j][0].item<int>();
            i2 = data[i][j][1].item<int>();
            fingerprint = data[i][j][2].item<int>();
            
            selected1 = torch::isin(filters.index({j, Slice(None, None), i1}), fingerprint).any().item<int>();
            
            if (selected1 == 1) {
                response[i] += (counters[j][i1].item<int>() > bleach);
            } else {
                selected2 = torch::isin(filters.index({j, Slice(None, None), i2}), fingerprint).any().item<int>();

                response[i] += ((counters[j][i2].item<int>() * selected2) > bleach);
            }
        }
    }

    return response;
}

/* Quotient Filter implementation:
 Paper: Don't thrash: how to cache your hash on flash <https://dl.acm.org/doi/10.14778/2350229.2350275>
*/
int BIT_OCCUPIEDS = 1; // metadata bit
int BIT_CONTINUATIONS = 2; // metadata bit
int BIT_SHIFTEDS = 4; // metadata bit
int BIT_EMPTY = 7; //metadata bit

int quotient_filter_right_shift(
    Tensor * filters, // filters: (n_neurons x filter_size)
    Tensor * metadata, // metadata: (n_neurons x filter_size)
    int neuron,
    int index    
) {
    int occupied;

    for (long int i = index + 1; i < filters->size(1); i++) {
        if ((metadata[0][neuron][i].item<int>() & BIT_EMPTY) == 0) {
            for (long int j = i - 1; j >= index && (i > 0); j--,i--) {
                occupied = metadata[0][neuron][i].item<int>() & BIT_OCCUPIEDS;
                metadata[0][neuron][i] = metadata[0][neuron][j];
                filters[0][neuron][i] = filters[0][neuron][j];

                if (occupied) {
                    metadata[0][neuron][i] = metadata[0][neuron][i].item<int>() | BIT_OCCUPIEDS;
                } else {
                    metadata[0][neuron][i] = metadata[0][neuron][i].item<int>() & ~BIT_OCCUPIEDS;
                }

                metadata[0][neuron][i] = metadata[0][neuron][i].item<int>() | BIT_SHIFTEDS;
            }

            return 1;
        }        
    }

    return 0;
}

int quotient_filter_right_shift_bleach(
    Tensor * filters, // filters: (n_neurons x filter_size)
    Tensor * metadata, // metadata: (n_neurons x filter_size)
    Tensor * counters, // counters: (n_neurons x filter_size)
    int neuron,
    int index    
) {
    int occupied;

    for (long int i = index + 1; i < filters->size(1); i++) {
        if ((metadata[0][neuron][i].item<int>() & BIT_EMPTY) == 0) {
            for (long int j = i - 1; j >= index && (i > 0); j--,i--) {
                occupied = metadata[0][neuron][i].item<int>() & BIT_OCCUPIEDS;
                metadata[0][neuron][i] = metadata[0][neuron][j];
                filters[0][neuron][i] = filters[0][neuron][j];
                counters[0][neuron][i] = counters[0][neuron][j];

                if (occupied) {
                    metadata[0][neuron][i] = metadata[0][neuron][i].item<int>() | BIT_OCCUPIEDS;
                } else {
                    metadata[0][neuron][i] = metadata[0][neuron][i].item<int>() & ~BIT_OCCUPIEDS;
                }

                metadata[0][neuron][i] = metadata[0][neuron][i].item<int>() | BIT_SHIFTEDS;
            }

            return 1;
        }        
    }

    return 0;
}

int quotient_filter_add(
    Tensor * filters, // filters: (n_neurons x filter_size)
    Tensor * metadata, // metadata: (n_neurons x filter_size)
    int neuron,
    int quotient,
    int remainder
) {

    int start_index = quotient;

    // Walk back to find the beginning of cluster
    while ((metadata[0][neuron][start_index].item<int>() & BIT_SHIFTEDS) > 0){
        start_index--;
    }
            
    // Walk forward to find the start of the run
    int run_index = start_index;
    while (start_index < quotient) {
        do{
            run_index++;
        } while ((run_index < filters->size(1)) && (((metadata[0][neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)));

        do{
            start_index++;
        } while ((start_index < run_index) && (((metadata[0][neuron][start_index].item<int>() & BIT_OCCUPIEDS) == 0)));

        if (run_index == filters->size(1)) {
            return 0;
        }
    }

    /* Found start of a run */

    // First case if slot is empty
    if ((metadata[0][neuron][run_index].item<int>() & BIT_EMPTY) == 0) {
        // Mark canonical slot as occupied
        metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;

        if (run_index != quotient) {
            metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_SHIFTEDS;
        }
        filters[0][neuron][run_index] = remainder;
        return 1;
    }
    
    // Check if the start of run has to be shifted. If so, then BIT_CONTINUATIONS is set in the shifted remainder.
    int old_remainder = filters[0][neuron][run_index].item<int>();

    if (remainder < old_remainder) {
        // Old remainder sets CONTINUATION
        metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_CONTINUATIONS;

        if (quotient_filter_right_shift(filters, metadata, neuron, run_index)) {
            // Mark canonical slot as occupied
            metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
            // New remainder unsets CONTINUATION
            metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() & ~BIT_CONTINUATIONS;            
            filters[0][neuron][run_index] = remainder;            
            return 1;
        } else {
            metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() & ~BIT_CONTINUATIONS;
            return 0;
        }

    } else if (remainder == old_remainder) {
        metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
        return 1;
    }

    run_index++;

    // Case that look the subsequent slots to insert new remainder in order.
    while ((run_index < filters->size(1)) && ((metadata[0][neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)){
        old_remainder = filters[0][neuron][run_index].item<int>();

        if (remainder < old_remainder) {
            if (quotient_filter_right_shift(filters, metadata, neuron, run_index)) {
                // Mark canonical slot as occupied
                metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
                filters[0][neuron][run_index] = remainder;            
                return 1;
            } else {
                return 0;
            }
        } else if (remainder == old_remainder) {
            metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
            return 1;
        }

        run_index++;
    }
            
    if (run_index == filters->size(1))
        return 0;
    
    // Case where was walked the whole run and not found appropriate slot.
    // A new slot will be added to the run. If the slot is full, it is shifted first.
    if (((metadata[0][neuron][run_index].item<int>() & BIT_EMPTY) == 0) || (quotient_filter_right_shift(filters, metadata, neuron, run_index))) {
        // Mark canonical slot as occupied
        metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
        metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_CONTINUATIONS;
        metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_SHIFTEDS;
        filters[0][neuron][run_index] = remainder; 
        return 1;
    }
    return 0;
}

int quotient_filter_add_bleach(
    Tensor * filters, // filters: (n_neurons x filter_size)
    Tensor * metadata, // metadata: (n_neurons x filter_size)
    Tensor * counters, // counters: (n_neurons x filter_size)
    int neuron,
    int quotient,
    int remainder
) {

    int start_index = quotient;

    // Walk back to find the beginning of cluster
    while ((metadata[0][neuron][start_index].item<int>() & BIT_SHIFTEDS) > 0){
        start_index--;
    }
            
    // Walk forward to find the start of the run
    int run_index = start_index;
    while (start_index < quotient) {
        do{
            run_index++;
        } while ((run_index < filters->size(1)) && (((metadata[0][neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)));

        do{
            start_index++;
        } while ((start_index < run_index) && (((metadata[0][neuron][start_index].item<int>() & BIT_OCCUPIEDS) == 0)));

        if (run_index == filters->size(1)) {
            return 0;
        }
    }

    /* Found start of a run */

    // First case if slot is empty
    if ((metadata[0][neuron][run_index].item<int>() & BIT_EMPTY) == 0) {
        // Mark canonical slot as occupied
        metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;

        if (run_index != quotient) {
            metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_SHIFTEDS;
        }
        filters[0][neuron][run_index] = remainder;
        counters[0][neuron][run_index] = counters[0][neuron][run_index].item<int>() + 1;
        return 1;
    }
    
    // Check if the start of run has to be shifted. If so, then BIT_CONTINUATIONS is set in the shifted remainder.
    int old_remainder = filters[0][neuron][run_index].item<int>();

    if (remainder < old_remainder) {
        // Old remainder sets CONTINUATION
        metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_CONTINUATIONS;

        if (quotient_filter_right_shift_bleach(filters, metadata, counters, neuron, run_index)) {
            // Mark canonical slot as occupied
            metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
            // New remainder unsets CONTINUATION
            metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() & ~BIT_CONTINUATIONS;            
            filters[0][neuron][run_index] = remainder; 
            counters[0][neuron][run_index] = counters[0][neuron][run_index].item<int>() + 1;            
            return 1;
        } else {
            metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() & ~BIT_CONTINUATIONS;
            return 0;
        }

    } else if (remainder == old_remainder) {
        metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
        return 1;
    }

    run_index++;

    // Case that look the subsequent slots to insert new remainder in order.
    while ((run_index < filters->size(1)) && ((metadata[0][neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)){
        old_remainder = filters[0][neuron][run_index].item<int>();

        if (remainder < old_remainder) {
            if (quotient_filter_right_shift_bleach(filters, metadata, counters, neuron, run_index)) {
                // Mark canonical slot as occupied
                metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
                filters[0][neuron][run_index] = remainder;  
                counters[0][neuron][run_index] = counters[0][neuron][run_index].item<int>() + 1;          
                return 1;
            } else {
                return 0;
            }
        } else if (remainder == old_remainder) {
            counters[0][neuron][run_index] = counters[0][neuron][run_index].item<int>() + 1;
            metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
            return 1;
        }

        run_index++;
    }
            
    if (run_index == filters->size(1))
        return 0;
    
    // Case where was walked the whole run and not found appropriate slot.
    // A new slot will be added to the run. If the slot is full, it is shifted first.
    if (((metadata[0][neuron][run_index].item<int>() & BIT_EMPTY) == 0) || (quotient_filter_right_shift(filters, metadata, neuron, run_index))) {
        // Mark canonical slot as occupied
        metadata[0][neuron][quotient] = metadata[0][neuron][quotient].item<int>() | BIT_OCCUPIEDS;
        metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_CONTINUATIONS;
        metadata[0][neuron][run_index] = metadata[0][neuron][run_index].item<int>() | BIT_SHIFTEDS;
        filters[0][neuron][run_index] = remainder; 
        counters[0][neuron][run_index] = counters[0][neuron][run_index].item<int>() + 1;
        return 1;
    }
    return 0;
}

int quotient_filter_check_member(
    Tensor & filters, // filters: (n_neurons x filter_size)
    Tensor & metadata, // metadata: (n_neurons x filter_size)
    int neuron,
    int quotient,
    int remainder
) {
    if (metadata[neuron][quotient].item<int>() & BIT_OCCUPIEDS) {
        int start_index = quotient;

        // Walk back to find the beginning of cluster
        while ((metadata[neuron][start_index].item<int>() & BIT_SHIFTEDS) > 0){
            start_index--;
        }
                
        // Walk forward to find the start of the run
        int run_index = start_index;
        while (start_index < quotient) {
            do{
                run_index++;
            } while ((run_index < filters.size(1)) && (((metadata[neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)));

            do{
                start_index++;
            } while ((start_index < run_index) && (((metadata[neuron][start_index].item<int>() & BIT_OCCUPIEDS) == 0)));

            if (run_index == filters.size(1)) {
                return 0;
            }
        }

        if (filters[neuron][run_index].item<int>() == remainder) 
            return 1;
            
        run_index++;

        while ((run_index < filters.size(1)) && ((metadata[neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)) {
            if (filters[neuron][run_index].item<int>() == remainder) 
                return 1;
            run_index++;
        }
    }

    return 0;
}

int quotient_filter_check_member_bleach(
    Tensor & filters, // filters: (n_neurons x filter_size)
    Tensor & metadata, // metadata: (n_neurons x filter_size)
    Tensor & counters, // counters: (n_neurons x filter_size)
    int neuron,
    int quotient,
    int remainder,
    int bleach // 0 - without bleach, > 0 - bleach value
) {
    if (metadata[neuron][quotient].item<int>() & BIT_OCCUPIEDS) {
        int start_index = quotient;

        // Walk back to find the beginning of cluster
        while ((metadata[neuron][start_index].item<int>() & BIT_SHIFTEDS) > 0){
            start_index--;
        }
                
        // Walk forward to find the start of the run
        int run_index = start_index;
        while (start_index < quotient) {
            do{
                run_index++;
            } while ((run_index < filters.size(1)) && (((metadata[neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)));

            do{
                start_index++;
            } while ((start_index < run_index) && (((metadata[neuron][start_index].item<int>() & BIT_OCCUPIEDS) == 0)));

            if (run_index == filters.size(1)) {
                return 0;
            }
        }

        if (filters[neuron][run_index].item<int>() == remainder) 
            return counters[neuron][run_index].item<int>() > bleach;
            
        run_index++;

        while ((run_index < filters.size(1)) && ((metadata[neuron][run_index].item<int>() & BIT_CONTINUATIONS) > 0)) {
            if (filters[neuron][run_index].item<int>() == remainder) 
                return counters[neuron][run_index].item<int>() > bleach;
            run_index++;
        }
    }

    return 0;
}

void quotient_filter_multi_add(
    Tensor * filters, // filters: (n_neurons x filter_size)
    Tensor * metadata, // metadata: (n_neurons x filter_size)
    Tensor & data, // data: (n_samples x n_filters), where each element is a fingerprint 
    int remainder_size    
) {   

    int quotient, remainder, remainder_mask = (1 << remainder_size) - 1;
    
    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            quotient = data[i][j].item<int>() >> remainder_size;
            remainder = data[i][j].item<int>() & remainder_mask;    
            quotient_filter_add(filters, metadata, j, quotient, remainder);
        }
    }
}

void quotient_filter_multi_add_bleach(
    Tensor * filters, // filters: (n_neurons x filter_size)
    Tensor * metadata, // metadata: (n_neurons x filter_size)
    Tensor * counters, // counters: (n_neurons x filter_size)
    Tensor & data, // data: (n_samples x n_filters), where each element is a fingerprint 
    int remainder_size    
) {   

    int quotient, remainder, remainder_mask = (1 << remainder_size) - 1;
    
    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            quotient = data[i][j].item<int>() >> remainder_size;
            remainder = data[i][j].item<int>() & remainder_mask;    
            quotient_filter_add_bleach(filters, metadata, counters, j, quotient, remainder);
        }
    }
}


torch::Tensor  quotient_filter_multi_rank(
    Tensor & filters, // filters: (n_neurons x filter_size)
    Tensor & metadata, // metadata: (n_neurons x filter_size)
    Tensor & data, // data: (n_samples x n_filters), where each element is a fingerprint 
    int remainder_size    
) {   

    int quotient, remainder, remainder_mask = (1 << remainder_size) - 1;

    const auto device = data.device();

    Tensor response = torch::zeros(
        {data.size(0)},
        torch::dtype(torch::kInt64).device(device));
    
    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            quotient = data[i][j].item<int>() >> remainder_size;
            remainder = data[i][j].item<int>() & remainder_mask;    
            response[i] += quotient_filter_check_member(filters, metadata, j, quotient, remainder);
        }
    }

    return response;
}

torch::Tensor  quotient_filter_multi_rank_bleach(
    Tensor & filters, // filters: (n_neurons x filter_size)
    Tensor & metadata, // metadata: (n_neurons x filter_size)
    Tensor & counters, // counters: (n_neurons x filter_size)
    Tensor & data, // data: (n_samples x n_filters), where each element is a fingerprint 
    int remainder_size,
    int bleach // 0 - without bleach, > 0 - bleach value        
) {   

    int quotient, remainder, remainder_mask = (1 << remainder_size) - 1;

    const auto device = data.device();

    Tensor response = torch::zeros(
        {data.size(0)},
        torch::dtype(torch::kInt64).device(device));
    
    for (long int i = 0; i < data.size(0); i++) {
        for (long int j = 0; j < data.size(1); j++) {
            quotient = data[i][j].item<int>() >> remainder_size;
            remainder = data[i][j].item<int>() & remainder_mask;    
            response[i] += quotient_filter_check_member_bleach(filters, metadata, counters, j, quotient, remainder, bleach);
        }
    }

    return response;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("h3_hash", &h3_hash, "Compute H3 hash function");
    m.def("h3_hash_int", &h3_hash_int, "Compute H3 hash function considering integer input");
    m.def("h3_multi_hash", &h3_multi_hash, "Compute H3 hash function for several inputs");
    m.def("h3_multi_hash_int", &h3_multi_hash_int, "Compute H3 hash function for several integer inputs");
    m.def("filter_multi_add", &filter_multi_add, "Add value for multiple filters");
    m.def("filter_multi_rank", &filter_multi_rank, "Calculate rank for multiple filters");
    m.def("cuckoo_filter_multi_add", &cuckoo_filter_multi_add, "Add value for multiple cuckoo filters");
    m.def("cuckoo_filter_multi_rank", &cuckoo_filter_multi_rank, "Calculate rank for multiple cuckoo filters");
    m.def("quotient_filter_multi_add", &quotient_filter_multi_add, "Add value for multiple quotient filters");
    m.def("quotient_filter_multi_rank", &quotient_filter_multi_rank, "Calculate rank for multiple quotient filters");
    m.def("cuckoo_filter_multi_add_bleach", &cuckoo_filter_multi_add_bleach, "Add value for multiple cuckoo filters using bleaching");
    m.def("cuckoo_filter_multi_rank_bleach", &cuckoo_filter_multi_rank_bleach, "Calculate rank for multiple cuckoo filters using bleaching");
    m.def("quotient_filter_multi_add_bleach", &quotient_filter_multi_add_bleach, "Add value for multiple quotient filters using bleaching");
    m.def("quotient_filter_multi_rank_bleach", &quotient_filter_multi_rank_bleach, "Calculate rank for multiple quotient filters using bleaching");
    m.def("sketch_multi_rank", &sketch_multi_rank, "Calculate rank for multiple sketch");
    
}
