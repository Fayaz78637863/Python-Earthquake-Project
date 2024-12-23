# Generated by Django 5.0.7 on 2024-08-06 09:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('adminapp', '0005_decision_algo_alter_ada_algo_table'),
    ]

    operations = [
        migrations.CreateModel(
            name='LOGISTIC_ALGO',
            fields=[
                ('S_NO', models.AutoField(primary_key=True, serialize=False)),
                ('Accuracy', models.TextField(max_length=100)),
                ('Precession', models.TextField(max_length=100)),
                ('F1_Score', models.TextField(max_length=100)),
                ('Recall', models.TextField(max_length=100)),
                ('Name', models.TextField(max_length=100)),
            ],
            options={
                'db_table': 'LOGISTIC_ALGO',
            },
        ),
    ]
