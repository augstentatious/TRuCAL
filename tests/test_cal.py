import torch
import torch.nn as nn
from cal import VulnerabilitySpotter, ConfessionalTemplate, TinyConfessionalLayer, CAL_TRM_Hybrid, ScratchpadLayer

# 1. Define d_model
d_model = 256
batch_size = 4
sequence_length = 10

print(f"Testing CAL-TRM modules with d_model={d_model}, batch_size={batch_size}, sequence_length={sequence_length}")
print("-" * 30)

# 2. Test VulnerabilitySpotter
print("Testing VulnerabilitySpotter...")
try:
    vs_input = torch.randn(batch_size, sequence_length, d_model)
    vulnerability_spotter = VulnerabilitySpotter(d_model)
    vs_output_v_t, vs_metadata = vulnerability_spotter(vs_input)

    if isinstance(vs_output_v_t, torch.Tensor):
        print(f"VulnerabilitySpotter v_t output shape: {vs_output_v_t.shape}")
    else:
        print(f"VulnerabilitySpotter v_t (scalar): {vs_output_v_t}")

    print(f"VulnerabilitySpotter metadata keys: {vs_metadata.keys()}")
    print("VulnerabilitySpotter test completed.")
except Exception as e:
    print(f"Error testing VulnerabilitySpotter: {e}")
print("-" * 30)

# 3. Test ConfessionalTemplate
print("Testing ConfessionalTemplate...")
try:
    ct_input = torch.randn(batch_size, sequence_length, d_model)
    confessional_template = ConfessionalTemplate(d_model)
    ct_output = confessional_template(ct_input, 'prior')
    print(f"ConfessionalTemplate output shape ('prior' template): {ct_output.shape}")
    ct_output_2 = confessional_template(ct_input, 'evidence')
    print(f"ConfessionalTemplate output shape ('evidence' template): {ct_output_2.shape}")
    print("ConfessionalTemplate test completed.")
except Exception as e:
    print(f"Error testing ConfessionalTemplate: {e}")
print("-" * 30)

# 4. Test TinyConfessionalLayer
print("Testing TinyConfessionalLayer...")
try:
    tcl_input = torch.randn(batch_size, sequence_length, d_model)
    tiny_confessional_layer = TinyConfessionalLayer(d_model)
    tcl_output, tcl_metadata = tiny_confessional_layer(tcl_input)
    print(f"TinyConfessionalLayer output shape: {tcl_output.shape}")
    print(f"TinyConfessionalLayer metadata keys: {tcl_metadata.keys()}")
    print("TinyConfessionalLayer test completed.")
except Exception as e:
    print(f"Error testing TinyConfessionalLayer: {e}")
print("-" * 30)

# 5. Test ScratchpadLayer
print("Testing ScratchpadLayer...")
try:
    sp_input = torch.randn(batch_size, sequence_length, d_model)
    scratchpad_layer = ScratchpadLayer(d_model)
    sp_output_initial = scratchpad_layer(sp_input)
    print(f"ScratchpadLayer initial output shape: {sp_output_initial.shape}")

    # Test with a previous state
    prev_z = torch.randn(batch_size, d_model)
    sp_output_with_prev = scratchpad_layer(sp_input, prev_z=prev_z)
    print(f"ScratchpadLayer output shape with previous state: {sp_output_with_prev.shape}")
    print("ScratchpadLayer test completed.")
except Exception as e:
    print(f"Error testing ScratchpadLayer: {e}")
print("-" * 30)

# 6. Test CAL_TRM_Hybrid (Main Test)
print("Testing CAL_TRM_Hybrid...")
try:
    hybrid_input = torch.randn(batch_size, sequence_length, d_model)
    hybrid_model = CAL_TRM_Hybrid(d_model=d_model)

    # Test forward pass
    hybrid_output, hybrid_metadata, scratchpad_state = hybrid_model(hybrid_input)
    print(f"CAL_TRM_Hybrid output shape: {hybrid_output.shape}")
    print(f"CAL_TRM_Hybrid metadata keys: {hybrid_metadata.keys()}")
    print(f"CAL_TRM_Hybrid scratchpad state shape: {scratchpad_state.shape}")
    print(f"Confessional triggered: {hybrid_metadata['confessional_triggered']}")

    print("CAL_TRM_Hybrid test completed.")
except Exception as e:
    print(f"Error testing CAL_TRM_Hybrid: {e}")
print("-" * 30)

print("All tests attempted.")
