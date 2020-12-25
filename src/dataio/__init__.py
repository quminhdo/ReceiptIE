from collections import Counter, defaultdict

def get_datasets(cf):
    text_files = sorted(glob(os.path.join(cf.ie_data_dir, "*.txt")))
    templates = Counter()
    Templates = defaultdict(list)

    for tf in text_files:
        with open(tf, "r") as f:
            d = json.load(f)
            Templates[d["company"]].append(re.sub("txt", "jpg", tf))
    companies = sorted(Templates.keys())

    n_train = int(len(companies) * cf.train_ratio)
    n_val = int(len(companies) * cf.val_ratio)
    n_test = len(companies) - n_train - n_val

    random.seed(0)
    random.shuffle(companies)
    train_templates = companies[:n_train]
    val_templates = companies[n_train : n_train+n_val]
    test_templates = companies[-n_test:]