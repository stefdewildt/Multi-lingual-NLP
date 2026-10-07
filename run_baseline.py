from detector.baseline.baseline_detector import train_detector


if __name__ == "__main__":
    dataset = 'multisocial'
    train_detector(dataset, save_dir=f'detector/baseline/models/{dataset}_all')
    train_detector(dataset, languages=['en'], save_dir=f'detector/baseline/models/{dataset}_en')
    train_detector(dataset, languages=['fr', 'de', 'en', 'es', 'pt'], save_dir=f'detector/baseline/models/{dataset}_big_eu')
    train_detector(dataset, languages=['nl', 'et', 'ga', 'gd'], save_dir=f'detector/baseline/models/{dataset}_less_common')

    dataset = 'multitude'
    train_detector(dataset, save_dir=f'detector/baseline/models/{dataset}_all')
    train_detector(dataset, languages=['en'], save_dir=f'detector/baseline/models/{dataset}_en')
    train_detector(dataset, languages=['fr', 'de', 'en', 'es', 'pt'], save_dir=f'detector/baseline/models/{dataset}_big_eu')
    train_detector(dataset, languages=['nl', 'et', 'ga', 'gd'], save_dir=f'detector/baseline/models/{dataset}_less_common')

    dataset = 'both'
    train_detector(dataset, save_dir=f'detector/baseline/models/{dataset}_all')
    train_detector(dataset, languages=['en'], save_dir=f'detector/baseline/models/{dataset}_en')
    train_detector(dataset, languages=['fr', 'de', 'en', 'es', 'pt'], save_dir=f'detector/baseline/models/{dataset}_big_eu')
    train_detector(dataset, languages=['nl', 'et', 'ga', 'gd'], save_dir=f'detector/baseline/models/{dataset}_less_common')