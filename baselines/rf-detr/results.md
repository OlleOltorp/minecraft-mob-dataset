# RF-DETR Large on the full dataset

First baseline: RF-DETR Large fine-tuned from COCO weights on the full dataset
(`datasets/full.toml`), settings in [config.toml](config.toml). Raw log: [metrics.csv](metrics.csv).

## Test split

5,583 pictures from tiles held out by place (the tiles furthest out).

| mAP50 | mAP50-95 | mAP75 | precision | recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.740 | 0.601 | 0.643 | 0.820 | 0.686 | 0.743 |

## Validation by epoch

| epoch | mAP50 | mAP50-95 | train loss |
|---:|---:|---:|---:|
| 1 | 0.520 | 0.422 | 7.47 |
| 2 | 0.652 | 0.526 | 5.87 |
| 3 | 0.697 | 0.560 | 5.35 |
| 4 | 0.718 | 0.578 | 5.01 |
| 5 | 0.730 | 0.587 | 4.72 |
| 6 | 0.735 | 0.592 | 4.46 |
| 7 | 0.740 | 0.596 | 4.26 |
| 8 | 0.744 | 0.598 | 4.08 |
| 9 | 0.743 | 0.597 | 3.96 |
| 10 | 0.742 | 0.597 | 3.91 |

## Test AP per class (AP50-95)

| class | AP | class | AP | class | AP |
|---|---:|---|---:|---|---:|
| magma_cube | 0.866 | bee | 0.650 | parrot | 0.554 |
| ender_dragon | 0.860 | rabbit | 0.650 | frog | 0.553 |
| ghast | 0.839 | fox | 0.649 | donkey | 0.552 |
| glow_squid | 0.811 | sheep | 0.647 | pillager | 0.548 |
| slime | 0.808 | skeleton_horse | 0.647 | skeleton | 0.546 |
| blaze | 0.808 | ravager | 0.647 | bogged | 0.536 |
| panda | 0.795 | goat | 0.644 | vindicator | 0.534 |
| allay | 0.776 | nautilus | 0.635 | horse | 0.523 |
| strider | 0.772 | chicken | 0.629 | wandering_trader | 0.514 |
| phantom | 0.771 | copper_golem | 0.628 | stray | 0.505 |
| cave_spider | 0.748 | warden | 0.622 | villager | 0.505 |
| pig | 0.745 | endermite | 0.616 | evoker | 0.502 |
| spider | 0.739 | breeze | 0.614 | wither | 0.500 |
| zombified_piglin | 0.736 | snow_golem | 0.599 | cod | 0.493 |
| shulker | 0.733 | witch | 0.596 | zombie_villager | 0.491 |
| armadillo | 0.722 | camel | 0.596 | llama | 0.473 |
| hoglin | 0.719 | cow | 0.593 | parched | 0.469 |
| sniffer | 0.713 | ocelot | 0.593 | wolf | 0.464 |
| bat | 0.711 | enderman | 0.591 | drowned | 0.463 |
| vex | 0.708 | mule | 0.587 | cat | 0.441 |
| zoglin | 0.698 | dolphin | 0.583 | iron_golem | 0.410 |
| zombie_horse | 0.679 | zombie | 0.581 | elder_guardian | 0.402 |
| piglin | 0.676 | polar_bear | 0.577 | wither_skeleton | 0.393 |
| creeper | 0.668 | player | 0.577 | zombie_nautilus | 0.339 |
| mooshroom | 0.667 | trader_llama | 0.573 | creaking | 0.319 |
| happy_ghast | 0.664 | camel_husk | 0.570 | tadpole | 0.285 |
| squid | 0.663 | piglin_brute | 0.569 | pufferfish | 0.245 |
| guardian | 0.662 | salmon | 0.564 | tropical_fish | 0.144 |
| turtle | 0.657 | silverfish | 0.559 |  |  |
| axolotl | 0.652 | husk | 0.555 |  |  |
