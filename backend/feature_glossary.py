FEATURE_GLOSSARY = {
    "TransactionAmt": "The transaction's payment amount, in USD.",
    "ProductCD": "A code representing the product category for the transaction (categories include W, C, R, H, S). Vesta did not publicly disclose exactly what each letter maps to in real-world terms — only that it distinguishes product types.",
    "card1": "An identifier associated with the payment card (e.g. card issuer/account identifier). Exact derivation not publicly disclosed by Vesta; used by the model as a numeric identifier signal, not a literal card number.",
    "card2": "A secondary payment card identifier field, same confidentiality caveat as card1.",
    "card3": "A card attribute field (e.g. country code). Exact derivation not publicly disclosed by Vesta.",
    "card4": "The card network, e.g. Visa, Mastercard, American Express, Discover.",
    "card5": "A card attribute field. Exact derivation not publicly disclosed by Vesta.",
    "card6": "The card type: credit or debit.",
    "card6_debit": "A one-hot encoded flag: 1 if card6 == 'debit', 0 otherwise (created during preprocessing, not an original dataset column).",
    "card6_credit": "A one-hot encoded flag: 1 if card6 == 'credit', 0 otherwise (created during preprocessing, not an original dataset column).",
    "addr1": "A billing address region field (e.g. associated with the cardholder's billing address). Exact geographic resolution not disclosed.",
    "addr2": "A secondary billing address field, same confidentiality caveat as addr1.",
    "dist1": "A distance-related feature (e.g. between billing and shipping address, or cardholder and transaction location). Exact unit/definition not publicly disclosed by Vesta.",
    "dist2": "A second distance-related feature, same confidentiality caveat as dist1.",
    "P_emaildomain": "The purchaser's email domain (e.g. gmail.com, yahoo.com).",
    "C_columns_general": "C1 through C14 are 'counting' features engineered by Vesta — for example, counts related to how many addresses, devices, or cards are associated with a given entity. Vesta did NOT publicly disclose the exact real-world meaning of each individual C column for confidentiality reasons; only that they are count-based signals.",
    "D_columns_general": "D1 through D15 are 'timedelta' features — differences in time (e.g. days) between the current transaction and some reference event (such as a previous transaction on the same card). Exact reference points for each individual D column are not publicly disclosed by Vesta.",
    "M_columns_general": "M1 through M9 are match flags — True/False/missing indicators of whether certain fields match (e.g. names on the card and address). Exact match criteria for each individual M column are not publicly disclosed by Vesta.",
    "isFraud": "The real target label: 1 if the transaction was identified as fraudulent, 0 otherwise.",
}


def lookup_feature(feature_name):
    """Exact match first, then fall back to the general C/D/M-column explanation."""
    if feature_name in FEATURE_GLOSSARY:
        return FEATURE_GLOSSARY[feature_name]
    if feature_name.startswith("C") and feature_name[1:].isdigit():
        return FEATURE_GLOSSARY["C_columns_general"]
    if feature_name.startswith("D") and feature_name[1:].isdigit():
        return FEATURE_GLOSSARY["D_columns_general"]
    if feature_name.startswith("M") and feature_name[1:].isdigit():
        return FEATURE_GLOSSARY["M_columns_general"]
    # Handle one-hot encoded features like ProductCD_W, card4_visa, etc.
    for prefix in ["ProductCD_", "card4_", "card6_", "P_emaildomain_"]:
        if feature_name.startswith(prefix):
            base = prefix.rstrip("_")
            if base in FEATURE_GLOSSARY:
                value = feature_name[len(prefix):]
                return f"{FEATURE_GLOSSARY[base]} (This is a one-hot flag for value '{value}'.)"
    return None
