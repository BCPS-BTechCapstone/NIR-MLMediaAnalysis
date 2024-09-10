# Experiment NIR Setup and Procedure

## Materials and Equipment

- DLP® NIRscan™ Nano Evaluation Module (Texas Instruments)
- TIDCC47 - DLP NIRscan Nano GUI v2.1.0 (Texas Instruments)
- Quartz Glass High Precision Cell (Hellma Analytics)
- PTFE Diffuse Reflector Sheet with Adhesive Backing (ThorLabs)
- Custom 3D Printed ABS Enclosure for NIR

## NIR Setup

### Cell Preparation

1) Rinse the Quartz Cell using DI water and a kimwipe
2) Rinse the Quartz Cell with 70% Ethanol
3) Rinse the Quartz Cell with Methanol
4) Use the vacuum port to suck out remaining moisture within the cell
5) Gently pat the exterior dry with a lint-free cloth

### NIR Software Setup

#### Setting Scan Reference

1) Turn on "Keep Lamp On"
2) Wait 5 minutes
3) Select `New` under Scan Reference Select
4) Place the Quartz Cell into the enclosure
5) Perform a scan
6) The Scan Reference Select will automatically change to `Previous`
7) The reference has now been set

#### Scan Parameters

Method: `4TR1 BCPS Def`

Filename Prefix: `Sample<num><type>_` (i.e. Sample3 or Sample3C)\
Folder: Folder with the same name as the Sample within 4TR3

Back-to-Back: `576`\
Scan Delay (s): `300`

### Starting the Scanning Process

1) Take the Quartz Cell out of the enclosure
2) Fill it with 3mL of the Sample
3) Carefully pipette 25mL of mineral oil on top of the sample to create a small layer.
4) Cover the Quartz Cell using parafilm
5) Place the Quartz Cell into the enclosure
6) Make sure all of the settings are correct within the software
7) Click "Scan" to start the scanning process
8) Ensure that the laptop is plugged in, then close the lid

