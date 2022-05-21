# BP-rozpoznavani-znakove-reci
Rozpoznávání datasetu LSA64 pomocí frameworku MMAction2. V tomto repozitáři jsou soubory použité k vypracování Bakalářské práce rozpoznávání jednotlivých znaků znakového jazyka a návod na jejich použití. Práce je zaměřena na rozpoznávání datasetu LSA64. Jako pre-requisite je využit framework MMAction2 https://github.com/open-mmlab/mmaction2.
# Použitý software
- python 3.7.9
- torch 1.8.0+cu101
- torchvision 0.9.0+cu101
- opencv 3.4.2
- mmcv-full 1.3.9
- mmaction2 0.20.0
- matplotlib 3.4.3

# Návod na použití
Pro použití je potřeba nainstalovat použitý software a přesunout soubory z tohoto repozitáře do složky MMAction2.

Dataset LSA64 stáhněte na stránce: http://facundoq.github.io/datasets/lsa64/. Ve složce data/lsa64/ vytvořte složky lsa64_train, lsa64_test, lsa64_val a vložte videa z datasetu do jedné z těchto složek. Poté spusťte data_distribution_setup_lsa64. Tento skript rozdělí data do složek tak, že do testovací množiny vloží všechna videa od jednoho figuranta určeného proměnnou CHOSEN_TEST_PERSON, ze zbytku vybere od každého figuranta jedno video určené proměnou CHOSEN_VAL_VID, které vloží do validační množiny, ostatní videa zůstanou v trénovací množině. Pro změnu figuranta na testování či videa pro validaci změňte hodnoty těchto proměnných 

Checkpoint předtrénovaný na datasetu kinetics400 stáhněte na stránce: https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth.

Pro spuštění trénování spusťte skript tools/train.py. V argumentech nastavte model, checkpoint, pracovní adresář, počet gpu. 
Například:
```shell
../checkpoints/checkpoint_lsa64/tsn_r50_video_1x1x8_100e_lsa64_rgb_sgd.py
--resume-from ../checkpoints/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth
--work-dir ../checkpoints/checkpoint_lsa64
--gpus 1
```

Pro spuštění testování spusťte skript tools/test.py. Před spuštěním se ale ujistěte, že máte stejně anotovaná videa, jako byla při trénování které vedlo k checkpointu.
Pokud ne, přenastavte ve skriptu data_distribution_setup_lsa64 proměnnou CHOSEN_TEST_PERSON na ID testovacího figuranta. V argumentech nastavte model, checkpoint, typ vyhodnocení
Například:
```shell
../checkpoints/checkpoint_lsa64/tsn_r50_video_1x1x8_100e_lsa64_rgb_sgd.py
../checkpoints/checkpoint_lsa64/latest.pth
--eval top_k_accuracy
```

Pro zobrazení grafů z trénování spusťte skript tools/train_resoults.py, ve kterém můžete nastavit v proměnné PATH cestu k json souboru z trénování. Pro tyto json soubory se můžete podívat na jejich využitý model v jejich stejnojmeném log souboru. Dále můžete nastavit PICKLE_DUMP na True, pokud chcete uložit křivky průběhu validace do pickle souborů. Tyto soubory jsou pak používány pro analýzu cross-validace.

Pro zobrazení grafů z cross-validace provedené v práci spusťte skript tools/cross_val_examination.py
