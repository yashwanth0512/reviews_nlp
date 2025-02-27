from bert_score import score

reference_summaries = {
    "Hot Wheels 48-Car Storage Case": """The Hot Wheels 48-Car Storage Case is a sturdy and practical solution for organizing Hot Wheels cars. 
    It features thick plastic construction, a well-spaced compartment system, and a securely painted-on design. 
    The case offers great value for money, keeping cars neatly stored and preventing scratches. 
    However, the vertical storage design lacks security to keep cars in place, and the case may not reliably fit 48 cars as advertised. 
    Some users also reported issues with build quality, including difficulties in closing and occasional breakage.""",
    
    "Baby Einstein Octoplush": """The Baby Einstein Octoplush is a durable and engaging educational toy that introduces children to colors and words in multiple languages. 
    It has a soft, baby-friendly design with vibrant colors and a high-quality build. The toy is designed to grow with the child and provides interactive learning. 
    However, some units suffer from durability issues, with the sound system malfunctioning after a short period. 
    Additionally, minor inconsistencies in color-word associations and the automatic "on" switch could cause confusion. 
    Despite these drawbacks, it remains a well-loved toy for early learning."""
}

generated_summaries = {
    "Hot Wheels 48-Car Storage Case": "The case is sturdy and a good solution for organizing cars. The plastic is thick, and it provides a decent way to store Hot Wheels. However, the vertical storage lacks security, and the advertised capacity is misleading. Some users faced closing difficulties and minor durability issues.",
    
    "Baby Einstein Octoplush": "This is a fun educational toy with a soft, durable design. It teaches colors and words in multiple languages and is engaging for babies. However, sound malfunctions occur in some cases, and the toy may become less exciting over time. Minor design issues include the on-switch being automatically turned on."
}

generated_list = list(generated_summaries.values())
reference_list = list(reference_summaries.values())

P, R, F1 = score(generated_list, reference_list, lang="en", rescale_with_baseline=True)


for i, (product, ref_summary) in enumerate(reference_summaries.items()):
    print(f"BERTScore for {product}:")
    print(f"Precision: {P[i].item():.4f}, Recall: {R[i].item():.4f}, F1-score: {F1[i].item():.4f}\n")
